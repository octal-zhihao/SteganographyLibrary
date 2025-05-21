# methods/neural_stego.py

import torch
import torch.nn.functional as F
from typing import ByteString, List
from utils import StegoMethod, load_causal_lm, limit_past, num_same_from_beg, bits2int, int2bits


def encode_arithmetic(model, enc, message_bits: List[int], context: List[int],
                      finish_sent: bool = False, device: str = 'cpu',
                      temp: float = 1.0, precision: int = 16, topk: int = 50000):
    """
    将 message_bits （0/1 列表）嵌入由 context 预热的模型中，返回生成的 token id 列表。
    """
    device = torch.device(device)
    context = torch.tensor(context[-1022:], device=device)
    max_val = 2**precision
    cur_interval = [0, max_val]
    prev = context
    past = None
    output = []

    i = 0
    sent_finish = False
    while i < len(message_bits) or (finish_sent and not sent_finish):
        out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
        logits = out.logits[0, -1].double()
        past = limit_past(out.past_key_values)

        # 屏蔽特殊 token
        logits[-1] = -1e20

        # 排序取 topk
        logits_sorted, indices = logits.sort(descending=True)
        logits_temp = logits_sorted / temp
        probs_temp = F.softmax(logits_temp, dim=0)

        # 计算当前区间长度与阈值
        cur_range = cur_interval[1] - cur_interval[0]
        threshold = 1 / cur_range
        k = min(max(2, (probs_temp < threshold).nonzero()[0].item()), topk)
        p_trunc = probs_temp[:k]
        p_trunc = (p_trunc / p_trunc.sum() * cur_range).round().long()
        cum = torch.cumsum(p_trunc, dim=0) + cur_interval[0]

        # 选择 message_bits 对应的 index
        bits_chunk = message_bits[i:i+precision]
        if len(bits_chunk) < precision:
            bits_chunk += [0] * (precision - len(bits_chunk))
        msg_idx = bits2int(list(reversed(bits_chunk)))
        selection = (cum > msg_idx).nonzero()[0].item()

        # 更新区间
        low = cum[selection-1].item() if selection > 0 else cur_interval[0]
        high = cum[selection].item()
        # 提取已固定的前缀位数
        low_bits = list(reversed(int2bits(low, precision)))
        high_bits = list(reversed(int2bits(high-1, precision)))
        nb = num_same_from_beg(low_bits, high_bits)
        i += nb
        # 生成 new interval
        new_low = bits2int(list(reversed(low_bits[nb:] + [0]*nb)))
        new_high = bits2int(list(reversed(high_bits[nb:] + [1]*nb))) + 1
        cur_interval = [new_low, new_high]

        # 选定 token
        tok_id = indices[selection].item()
        output.append(tok_id)

        prev = torch.tensor([tok_id], device=device)
        # 提前结束句子
        if finish_sent:
            text = enc.decode(output)
            if '<eos>' in text:
                break

    return output

def decode_arithmetic(model, enc, text: str, context: List[int],
                      device: str = 'cpu', temp: float = 1.0,
                      precision: int = 16, topk: int = 50000) -> List[int]:
    """
    从 stego 文本中逐 token 解码，返回恢复的 message_bits 列表。
    """
    device = torch.device(device)
    inp = enc.encode(text)
    # 处理 BPE 小错误（同官方）
    i = 0
    while i < len(inp):
        if inp[i] == 628:
            inp[i] = 198
            inp[i+1:i+1] = [198]
            i += 2
        else:
            i += 1

    context = torch.tensor(context[-1022:], device=device)
    cur_interval = [0, 2**precision]
    prev = context
    past = None
    message: List[int] = []

    i = 0
    for tok in inp:
        out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
        logits = out.logits[0, -1].double()
        past = limit_past(out.past_key_values)
        logits[-1] = -1e20

        logits_sorted, indices = logits.sort(descending=True)
        logits_temp = logits_sorted / temp
        probs_temp = F.softmax(logits_temp, dim=0)

        cur_range = cur_interval[1] - cur_interval[0]
        threshold = 1 / cur_range
        k = min(max(2, (probs_temp < threshold).nonzero()[0].item()), topk)
        p_trunc = probs_temp[:k]
        p_trunc = (p_trunc / p_trunc.sum() * cur_range).round().long()
        cum = torch.cumsum(p_trunc, dim=0) + cur_interval[0]

        # 找到 tok 在 indices 中的位置
        rank = (indices[:k] == tok).nonzero()
        if len(rank) == 0:
            break
        rank = rank[0].item()

        low = cum[rank-1].item() if rank>0 else cur_interval[0]
        high = cum[rank].item()
        low_bits = list(reversed(int2bits(low, precision)))
        high_bits = list(reversed(int2bits(high-1, precision)))
        nb = num_same_from_beg(low_bits, high_bits)
        # 提取这段共享位
        bits = high_bits[:nb]
        message += bits

        new_low = bits2int(list(reversed(low_bits[nb:] + [0]*nb)))
        new_high = bits2int(list(reversed(high_bits[nb:] + [1]*nb))) + 1
        cur_interval = [new_low, new_high]

        prev = torch.tensor([tok], device=device)
        i += 1

    return message


class NeuralStego(StegoMethod):
    def __init__(
        self,
        model_name: str = 'gpt2',
        temp: float = 1.0,
        precision: int = 16,
        topk: int = 50000,
        device: str = 'cpu'
    ):
        """
        基于算术编码的神经语言隐写
        temp: softmax 温度
        precision: 算术编码精度（bit-length）
        topk: 采样时截断到 topk 个 token
        """
        self.tokenizer, self.model = load_causal_lm(model_name)
        self.temp = temp
        self.precision = precision
        self.topk = topk
        self.device = device
        self.model.to(device)

    def encrypt(self, cover: str, payload: ByteString) -> str:
        # 1. 构造消息 bit 列表，加入 32-bit header
        msg = payload.decode('utf-8', errors='ignore')
        msg_bits = [int(b) for b in ''.join(f"{ord(c):016b}" for c in msg)]
        header = [int(b) for b in format(len(msg_bits), '032b')]
        full_bits = header + msg_bits

        # 2. 准备 context
        cover_ids = self.tokenizer.encode(cover, add_special_tokens=False)
        # 3. arithmetic encode → token ids
        stego_ids = encode_arithmetic(
            self.model, self.tokenizer, full_bits, cover_ids,
            finish_sent=True,
            device=self.device,
            temp=self.temp,
            precision=self.precision,
            topk=self.topk
        )
        # 4. decode 返回文本
        return self.tokenizer.decode(stego_ids, skip_special_tokens=True)

    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        # 1. split context
        cover_ids = self.tokenizer.encode(cover, add_special_tokens=False)
        # 2. arithmetic decode → bit 列表
        bits = decode_arithmetic(
            self.model, self.tokenizer,
            stego_text, cover_ids,
            device=self.device,
            temp=self.temp,
            precision=self.precision,
            topk=self.topk
        )
        # 3. 取前 32-bit 作为长度，再恢复消息
        header = bits[:32]
        length = bits2int(header)
        payload_bits = bits[32:32+length]
        # 4. 按 16-bit 一组还原字符
        chars = []
        for i in range(0, len(payload_bits), 16):
            val = bits2int(payload_bits[i:i+16])
            chars.append(chr(val))
        return ''.join(chars).encode('utf-8')
