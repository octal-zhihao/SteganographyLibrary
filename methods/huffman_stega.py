# methods/huffman_stega.py

import torch
import torch.nn.functional as F
from typing import ByteString, List
from methods.huffman import HuffmanCoding
from utils import StegoMethod, load_causal_lm, limit_past

class HuffmanStega(StegoMethod):
    def __init__(
        self,
        model_name: str      = 'gpt2',
        bits_per_word: int   = 2,
        encoding_method: str = 'flc',
        device: str          = 'cpu'
    ):
        """
        bits_per_word: 每词嵌入比特数 k；定长编码时 m = 2**k，变长同样用前 2**k 候选构树。
        encoding_method: 'flc'（Fixed-Length）或 'vlc'（Variable-Length Huffman）。
        """
        self.tokenizer, self.model = load_causal_lm(model_name)
        self.k               = bits_per_word
        self.top_m           = 2 ** bits_per_word
        self.encoding_method = encoding_method.lower()
        self.device          = device
        self.model.to(self.device)

    def encrypt(self, cover: str, payload: ByteString) -> str:
        # 1. 构造 bit 流
        bitstr = ''.join(f"{b:08b}" for b in payload)
        L      = len(bitstr)
        i      = 0

        # 2. 通过 encode 得到 cover_ids，并预热 past
        cover_ids = self.tokenizer.encode(cover, return_tensors='pt')[0].to(self.device)
        with torch.no_grad():
            out  = self.model(cover_ids.unsqueeze(0), use_cache=True)
            past = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)
        last_id = cover_ids[-1].view(1, 1)

        # 3. 隐写生成阶段
        suffix_ids: List[int] = []
        with torch.no_grad():
            while i < L:
                out    = self.model(last_id, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]
                past   = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)

                # 拿更多几个 top 以便过滤
                top_logits, top_ids = logits.topk(self.top_m * 2)
                top_probs = F.softmax(top_logits, dim=-1)

                # 过滤低价值标点，取最初 top_m
                candidates: List[int] = []
                for tid in top_ids.tolist():
                    tok = self.tokenizer.convert_ids_to_tokens(tid)
                    if tok in {',',';',':','"','“','”'}:
                        continue
                    candidates.append(tid)
                    if len(candidates) >= self.top_m:
                        break

                # 若不足，则回退至不过滤
                if len(candidates) < 2:
                    candidates = top_ids[:self.top_m].tolist()

                # 选择 token
                if self.encoding_method == 'flc':
                    chunk    = bitstr[i : i + self.k].ljust(self.k, '0')
                    idx      = int(chunk, 2)
                    token_id = candidates[idx]
                    i       += self.k
                else:  # vlc
                    probs = top_probs[:len(candidates)].cpu().numpy()
                    h = HuffmanCoding()
                    h.make_heap_from_array(probs)
                    h.merge_nodes()
                    root = h.make_codes()
                    node = root
                    while node.token is None and i < L:
                        node = node.right if bitstr[i] == '1' else node.left
                        i += 1
                        if node is None:
                            node = root
                            break
                    rank     = node.token or 0
                    token_id = candidates[rank]

                suffix_ids.append(token_id)
                last_id = torch.tensor([[token_id]], device=self.device)

            # 4. 补齐当前句子：贪心生成到句尾
            while True:
                out    = self.model(last_id, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]
                past   = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)
                token_id = torch.argmax(logits).item()
                suffix_ids.append(token_id)
                last_id = torch.tensor([[token_id]], device=self.device)
                tok     = self.tokenizer.convert_ids_to_tokens(token_id)
                if tok in {'.', '!', '?'}:
                    break

        # 5. 合并 cover + suffix 并 decode 返回
        full_ids = torch.cat([
            cover_ids,
            torch.tensor(suffix_ids, device=self.device)
        ])
        return self.tokenizer.decode(full_ids.tolist(), skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        # 1. 用 encode 切分出 suffix_ids
        cover_ids = self.tokenizer.encode(cover)
        stego_ids = self.tokenizer.encode(stego_text)
        suffix_ids = stego_ids[len(cover_ids):]

        # 2. 重新预热 cover
        ids_tensor = torch.tensor(cover_ids, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out  = self.model(ids_tensor, use_cache=True)
            past = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)
        last_id = torch.tensor([[cover_ids[-1]]], device=self.device)

        # 3. 提取比特
        bits: List[str] = []
        with torch.no_grad():
            for tid in suffix_ids:
                out    = self.model(last_id, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]
                past   = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)

                # 同样过滤标点
                top_logits, top_ids = logits.topk(self.top_m * 2)
                top_probs = F.softmax(top_logits, dim=-1)

                candidates: List[int] = []
                for cand in top_ids.tolist():
                    tok = self.tokenizer.convert_ids_to_tokens(cand)
                    if tok in {',',';',':','"','“','”'}:
                        continue
                    candidates.append(cand)
                    if len(candidates) >= self.top_m:
                        break
                if len(candidates) < 2:
                    candidates = top_ids[:self.top_m].tolist()

                top_list = candidates

                if tid not in top_list:
                    break
                rank = top_list.index(tid)

                if self.encoding_method == 'flc':
                    bits.append(format(rank, f'0{self.k}b'))
                else:
                    probs = F.softmax(top_logits[:len(candidates)], dim=-1)[:len(candidates)].cpu().numpy()
                    h = HuffmanCoding()
                    h.make_heap_from_array(probs)
                    h.merge_nodes()
                    _ = h.make_codes()
                    bits.append(h.codes[rank])

                last_id = torch.tensor([[tid]], device=self.device)

        # 4. 拼接比特串并转 bytes
        bitstr = ''.join(bits)
        valid  = (len(bitstr) // 8) * 8
        bitstr = bitstr[:valid]
        if not bitstr:
            return b''

        val  = int(bitstr, 2)
        return val.to_bytes(len(bitstr)//8, byteorder='big')
