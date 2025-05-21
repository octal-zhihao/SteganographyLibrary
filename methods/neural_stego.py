import torch
import torch.nn.functional as F
from typing import ByteString, List

from utils import StegoMethod, load_causal_lm, limit_past


def encode_arithmetic_fp(
    model, enc,
    message_bits: List[int],
    context_ids: List[int],
    device: str,
    temp: float,
    topk: int
) -> List[int]:
    """
    极简浮点算术编码：只为每个 bit 生成一个 token，1-to-1 映射，不做句末补齐。
    """
    device = torch.device(device)
    # 1. 预热 context
    prev_ids = torch.tensor(context_ids[-1022:], device=device).unsqueeze(0)
    past = None

    out_ids: List[int] = []
    low, high = 0.0, 1.0
    bit_ptr = 0
    total_bits = len(message_bits)

    with torch.no_grad():
        while bit_ptr < total_bits:
            outputs = model(prev_ids, past_key_values=past, use_cache=True)
            logits = outputs.logits[0, -1]       # [vocab_size]
            past = limit_past(outputs.past_key_values)

            # 屏蔽 eos
            logits[..., -1] = -1e9

            # 拿 topk
            top_logits, top_ids = logits.topk(topk)
            probs = F.softmax(top_logits / temp, dim=-1).cpu().numpy()
            top_ids = top_ids.cpu().numpy()

            # 构造累积概率 [0,1]
            cum = probs.cumsum()
            cum = cum / cum[-1]

            # 消耗一个 bit：二分 [low,high)
            b = message_bits[bit_ptr]
            mid = (low + high) / 2
            if b == 0:
                high = mid
            else:
                low = mid
            bit_ptr += 1

            # 选 token：找到第一个 cum > low
            idx = int((cum > low).argmax())
            tok = int(top_ids[idx])

            out_ids.append(tok)
            prev_ids = torch.tensor([[tok]], device=device)

    return out_ids


def decode_arithmetic_fp(
    model, enc,
    stego_ids: List[int],
    context_ids: List[int],
    total_bits: int,
    device: str,
    temp: float,
    topk: int
) -> List[int]:
    """
    极简浮点解码：重走 encode 时的逻辑，为了每个 token 反推一个 bit。
    """
    device = torch.device(device)
    prev_ids = torch.tensor(context_ids[-1022:], device=device).unsqueeze(0)
    past = None

    low, high = 0.0, 1.0
    recovered: List[int] = []

    with torch.no_grad():
        for tok in stego_ids:
            if len(recovered) >= total_bits:
                break

            outputs = model(prev_ids, past_key_values=past, use_cache=True)
            logits = outputs.logits[0, -1]
            past = limit_past(outputs.past_key_values)

            logits[..., -1] = -1e9
            top_logits, top_ids = logits.topk(topk)
            probs = F.softmax(top_logits / temp, dim=-1).cpu().numpy()
            top_ids = top_ids.cpu().numpy()

            cum = probs.cumsum()
            cum = cum / cum[-1]

            # 找到 tok 在 top_ids 中的位置
            ranks = (top_ids == tok).nonzero()
            if len(ranks) == 0:
                # 一旦遇到不在 topk 的 token，就中断
                break
            idx = int(ranks[0])

            # cum[idx] 是 token 在 [0,1) 的上界
            # 二分区间 mid
            mid = (low + high) / 2
            # 如果 cum[idx] ≤ mid，说明它落在左半区 => bit=0，否则 bit=1
            bit = 0 if cum[idx] <= mid else 1
            recovered.append(bit)

            # 更新浮点区间
            if bit == 0:
                high = mid
            else:
                low = mid

            prev_ids = torch.tensor([[tok]], device=device)

    return recovered


class NeuralStego(StegoMethod):
    def __init__(self,
                 model_name: str = 'gpt2',
                 temp: float = 1.0,
                 topk: int = 512,
                 device: str = 'cpu'):
        """
        极简浮点算术隐写：1bit→1token，不做句末补齐。
        """
        self.tokenizer, self.model = load_causal_lm(model_name)
        self.temp = temp
        self.topk = topk
        self.device = device
        self.model.to(device)

    def encrypt(self, cover: str, payload: ByteString) -> str:
        # bits 列表（8bit per byte）+ 32bit header
        bits: List[int] = []
        for b in payload:
            bits.extend(int(x) for x in format(b, '08b'))
        header = [int(x) for x in format(len(bits), '032b')]
        full_bits = header + bits

        # 预热 context
        context_ids = self.tokenizer.encode(cover, add_special_tokens=False)

        # 算术编码
        stego_ids = encode_arithmetic_fp(
            self.model, self.tokenizer,
            full_bits, context_ids,
            device=self.device,
            temp=self.temp,
            topk=self.topk
        )

        # 只输出隐写段
        return self.tokenizer.decode(stego_ids, skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        # 预热 context 与隐写 ids
        context_ids = self.tokenizer.encode(cover, add_special_tokens=False)
        stego_ids   = self.tokenizer.encode(stego_text, add_special_tokens=False)

        # 先解 header（32 bit）
        header_bits = decode_arithmetic_fp(
            self.model, self.tokenizer,
            stego_ids, context_ids,
            total_bits=32,
            device=self.device,
            temp=self.temp,
            topk=self.topk
        )
        length = int(''.join(str(b) for b in header_bits), 2)

        # 再解后续 payload bits
        payload_bits = decode_arithmetic_fp(
            self.model, self.tokenizer,
            stego_ids, context_ids,
            total_bits=32 + length,
            device=self.device,
            temp=self.temp,
            topk=self.topk
        )[32:]

        # 重建 bytes
        out = bytearray()
        for i in range(0, len(payload_bits), 8):
            chunk = payload_bits[i:i+8]
            if len(chunk) < 8:
                chunk += [0]*(8-len(chunk))
            out.append(int(''.join(str(b) for b in chunk), 2))
        return bytes(out)
