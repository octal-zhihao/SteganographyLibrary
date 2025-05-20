from typing import List, ByteString
from io import StringIO
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from utils import StegoMethod, load_masked_lm

class EditStego(StegoMethod):
    def __init__(
        self,
        model_name: str = 'bert-base-cased',
        mask_interval: int = 2,
        score_threshold: float = 0.01
    ):
        self.tokenizer, self.model = load_masked_lm(model_name)
        self.mask_interval = mask_interval
        self.score_threshold = score_threshold
        self.stopwords = set(stopwords.words('english'))

    def encrypt(self, cover: str, payload: ByteString) -> str:
        # 将 payload 转为比特串
        bits = ''.join(f"{byte:08b}" for byte in payload)
        message_io = StringIO(bits)

        # Tokenize 并做掩码
        tokens   = self.tokenizer.tokenize(cover)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        masked_ids = self._mask(input_ids.clone())
        orig_mask  = masked_ids.clone()

        # 预测概率分布
        sorted_score, indices = self._predict(masked_ids)

        # 遍历所有掩码位置，嵌入比特
        for idx in range(len(masked_ids)):
            if masked_ids[idx] != self.tokenizer.mask_token_id:
                continue
            candidates = self._pick_candidates(indices[idx], sorted_score[idx])
            if len(candidates) < 2:
                continue
            replace_id = self._block_encode_single(candidates, message_io)
            input_ids[idx] = replace_id

        # 统计真正被替换的位置数
        mask_id = self.tokenizer.mask_token_id
        embedded_positions = int(((orig_mask == mask_id) & (input_ids != mask_id)).sum().item())

        new_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        stego = self.tokenizer.convert_tokens_to_string(new_tokens)
        print(f'[DEBUG] message bits: {len(bits)}, embedded positions: {embedded_positions}')
        return stego

    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        message_bits = []

        # Tokenize 并做同样的掩码
        tokens    = self.tokenizer.tokenize(stego_text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        masked_ids = self._mask(input_ids.clone())

        # 预测概率分布
        sorted_score, indices = self._predict(masked_ids)

        # 遍历掩码位置，提取比特
        for idx in range(len(masked_ids)):
            if masked_ids[idx] != self.tokenizer.mask_token_id:
                continue
            candidates = self._pick_candidates(indices[idx], sorted_score[idx])
            if len(candidates) < 2:
                continue
            chosen = input_ids[idx].item()
            bits = self._block_decode_single(candidates, chosen)
            message_bits.append(bits)

        # 拼接 bitstr 并补齐到 8 的倍数
        bitstr = ''.join(message_bits)
        if len(bitstr) % 8 != 0:
            bitstr += '0' * (8 - len(bitstr) % 8)
        byts = int(bitstr, 2).to_bytes(len(bitstr) // 8, byteorder='big', signed=False)
        return byts

    def _mask(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
        length = len(ids)
        count = 0
        for i, tok in enumerate(tokens):
            # 跳过子词
            if i + 1 < length and tok.startswith('##'):
                continue
            # 跳过停用词与非字母
            if tok.lower() in self.stopwords or not tok.isalpha():
                continue
            # 每隔 mask_interval 掩码一次
            if count % self.mask_interval == 0:
                ids[i] = self.tokenizer.mask_token_id
            count += 1
        return ids

    def _predict(self, ids: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ids.unsqueeze(0))['logits'][0]
            return logits.sort(dim=-1, descending=True)

    def _pick_candidates(
        self,
        id_tensor: torch.Tensor,
        score_tensor: torch.Tensor
    ) -> List[int]:
        filtered = []
        # 只取 top-k，保证顺序稳定
        for idx in id_tensor.tolist():
            tok = self.tokenizer.convert_ids_to_tokens(idx)
            if tok.startswith('##') or tok.lower() in self.stopwords or not tok.isalpha():
                continue
            filtered.append(idx)
            if len(filtered) >= 16:
                break
        return filtered

    @staticmethod
    def _block_encode_single(ids: List[int], message_io: StringIO) -> int:
        k = len(ids)
        # 用 (k-1).bit_length() 确定能编码的比特数
        n = (k - 1).bit_length()
        if n <= 0:
            return ids[0]
        bits = message_io.read(n)
        if len(bits) < n:
            bits += '0' * (n - len(bits))
        idx = int(bits, 2)
        idx = min(idx, k - 1)
        return ids[idx]

    @staticmethod
    def _block_decode_single(ids: List[int], chosen: int) -> str:
        if len(ids) < 2:
            return ''
        n = (len(ids) - 1).bit_length()
        try:
            idx = ids.index(chosen)
        except ValueError:
            return ''
        return format(idx, f'0{n}b')
