import random
from typing import ByteString, List
import torch
from torch import Tensor
import torch.nn.functional as F
from utils import StegoMethod, load_causal_lm, limit_past

class LSTMStego(StegoMethod):
    def __init__(
        self,
        model_name: str = 'gpt2',
        bit_block_size: int = 2,
        seed: int = 42,
        device: str = 'cpu'
    ):
        """
        bit_block_size: B，比特块长度
        seed: 随机种子，用于构造固定的桶划分
        """
        self.tokenizer, self.model = load_causal_lm(model_name)
        self.B = bit_block_size
        self.seed = seed
        self.device = device
        self.model.to(self.device)

        # 构建桶 (bins)
        vocab_size = len(self.tokenizer)
        all_ids = [i for i in range(vocab_size) if i not in self.tokenizer.all_special_ids]
        rnd = random.Random(self.seed)
        rnd.shuffle(all_ids)
        num_bins = 2 ** self.B
        base = len(all_ids) // num_bins
        self.bins: List[List[int]] = []
        for i in range(num_bins):
            start = i * base
            end = (i + 1) * base if i < num_bins - 1 else len(all_ids)
            self.bins.append(all_ids[start:end])
        # 反向映射
        self.id2bits = {tid: format(idx, f'0{self.B}b')
                        for idx, bucket in enumerate(self.bins)
                        for tid in bucket}

    def encrypt(self, cover: str, payload: ByteString) -> str:
        # 1. bit 流分块
        bitstr = ''.join(f"{b:08b}" for b in payload)
        blocks = [bitstr[i:i+self.B].ljust(self.B, '0')
                  for i in range(0, len(bitstr), self.B)]

        # 2. 预热 cover，获取 past 和 last_id
        cover_ids = self.tokenizer.encode(cover, return_tensors='pt')[0].to(self.device)
        with torch.no_grad():
            out = self.model(cover_ids.unsqueeze(0), use_cache=True)
        past = limit_past(out.past_key_values,
                          max_length=self.model.config.n_positions-1)
        last_id = cover_ids[-1].view(1,1)

        # 3. 增量生成每块
        suffix_ids: List[int] = []
        with torch.no_grad():
            for block in blocks:
                bin_idx = int(block, 2)
                bin_ids = set(self.bins[bin_idx])

                # 增量调用
                out = self.model(last_id, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]  # shape [vocab]
                past = limit_past(out.past_key_values,
                                  max_length=self.model.config.n_positions-1)

                # 掩码不在桶的
                mask = torch.full_like(logits, float('-inf'))
                mask[list(bin_ids)] = logits[list(bin_ids)]
                next_id = torch.argmax(mask).item()

                suffix_ids.append(next_id)
                last_id = torch.tensor([[next_id]], device=self.device)

        # 4. decode 返回续写文本
        return self.tokenizer.decode(suffix_ids, skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        cover_ids = self.tokenizer.encode(cover)
        stego_ids = self.tokenizer.encode(stego_text)
        suffix_ids = stego_ids

        # 2. 预热 cover
        if cover_ids:
            ids_tensor = torch.tensor(cover_ids, device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(ids_tensor, use_cache=True)
            past = limit_past(out.past_key_values,
                              max_length=self.model.config.n_positions-1)
            last_id = torch.tensor([[cover_ids[-1]]], device=self.device)
        else:
            past = None
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            last_id = torch.tensor([[bos]], device=self.device)

        # 3. 增量提取 bits
        bits: List[str] = []
        with torch.no_grad():
            for tid in suffix_ids:
                out = self.model(last_id, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]
                past = limit_past(out.past_key_values,
                                  max_length=self.model.config.n_positions-1)
                # 直接用 id2bits
                bits.append(self.id2bits.get(tid, ''))
                last_id = torch.tensor([[tid]], device=self.device)

        bitstr = ''.join(bits)
        valid = (len(bitstr)//8)*8
        bitstr = bitstr[:valid]
        if not bitstr:
            return b''
        val = int(bitstr, 2)
        return val.to_bytes(valid//8, byteorder='big')
