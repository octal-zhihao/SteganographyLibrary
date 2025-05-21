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

        # 2. 用 cover 预热 past（仅用于内部上下文）
        cover_ids = self.tokenizer.encode(cover, return_tensors='pt')[0].to(self.device)
        with torch.no_grad():
            out  = self.model(cover_ids.unsqueeze(0), use_cache=True)
            past = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)
        last_id = cover_ids[-1].view(1, 1)

        # 3. 隐写生成：仅保存 suffix_ids
        suffix_ids: List[int] = []
        with torch.no_grad():
            # 嵌入比特
            while i < L:
                out    = self.model(last_id, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]
                past   = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)

                # 取更多候选以便过滤
                top_logits, top_ids = logits.topk(self.top_m * 2)
                top_probs = F.softmax(top_logits, dim=-1)

                # 过滤低价值标点，取前 top_m
                candidates: List[int] = []
                for tid in top_ids.tolist():
                    tok = self.tokenizer.convert_ids_to_tokens(tid)
                    if tok in {',',';',':','"','“','”'}:
                        continue
                    candidates.append(tid)
                    if len(candidates) >= self.top_m:
                        break
                if len(candidates) < 2:
                    candidates = top_ids[:self.top_m].tolist()

                # 按 flc/vlc 选词
                if self.encoding_method == 'flc':
                    chunk    = bitstr[i : i + self.k].ljust(self.k, '0')
                    idx      = int(chunk, 2)
                    token_id = candidates[idx]
                    i       += self.k
                else:
                    probs = top_probs[:len(candidates)].cpu().numpy()
                    h = HuffmanCoding(); h.make_heap_from_array(probs); h.merge_nodes()
                    root = h.make_codes(); node = root
                    while node.token is None and i < L:
                        node = node.right if bitstr[i]=='1' else node.left
                        i += 1
                        if node is None:
                            node = root; break
                    rank     = node.token or 0
                    token_id = candidates[rank]

                suffix_ids.append(token_id)
                last_id = torch.tensor([[token_id]], device=self.device)

            # 补齐当前句子直到遇到句尾符号
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

        # 4. 只 decode suffix 返回，不保留 cover
        return self.tokenizer.decode(suffix_ids, skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        # 1. stego_text 全部当作 suffix_ids
        suffix_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)

        # 2. 用 cover 预热 past
        cover_ids  = self.tokenizer.encode(cover, return_tensors='pt')[0].to(self.device)
        with torch.no_grad():
            out  = self.model(cover_ids.unsqueeze(0), use_cache=True)
            past = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)
        last_id = cover_ids[-1].view(1, 1)

        # 3. 逐词提取
        bits: List[str] = []
        with torch.no_grad():
            for tid in suffix_ids:
                out    = self.model(last_id, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]
                past   = limit_past(out.past_key_values, max_length=self.model.config.n_positions-1)

                top_logits, top_ids = logits.topk(self.top_m * 2)
                top_probs = F.softmax(top_logits, dim=-1)

                # 重复过滤逻辑
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

                if tid not in candidates:
                    break
                rank = candidates.index(tid)

                if self.encoding_method == 'flc':
                    bits.append(format(rank, f'0{self.k}b'))
                else:
                    probs = top_probs[:len(candidates)].cpu().numpy()
                    h = HuffmanCoding(); h.make_heap_from_array(probs); h.merge_nodes()
                    _ = h.make_codes()
                    bits.append(h.codes[rank])

                last_id = torch.tensor([[tid]], device=self.device)

        # 4. 拼接比特并转 bytes
        bitstr = ''.join(bits)
        valid  = (len(bitstr)//8)*8
        bitstr = bitstr[:valid]
        if not bitstr:
            return b''

        val  = int(bitstr, 2)
        return val.to_bytes(len(bitstr)//8, byteorder='big')
