import random
from typing import ByteString, List
import torch
from torch import Tensor
import torch.nn.functional as F
from utils import StegoMethod, load_causal_lm

class LSTMStego(StegoMethod):
    def __init__(
        self,
        model_name: str = 'gpt2',
        bit_block_size: int = 2,
        seed: int = 42
    ):
        """
        model_name: HuggingFace 上的因果语言模型名称
        bit_block_size: B，论文中的比特块长度
        seed: 随机种子，用于构造共享分桶键
        """
        # 加载因果 LM
        self.tokenizer, self.model = load_causal_lm(model_name)
        self.B = bit_block_size
        self.seed = seed

        # 构建分桶：将 tokenizer.vocab 随机分成 2**B 个桶
        vocab_size = len(self.tokenizer)
        # 只使用普通 token id（排除特殊 token）
        all_ids = [
            idx for idx in range(vocab_size)
            if idx not in self.tokenizer.all_special_ids
        ]
        # 打乱顺序
        rnd = random.Random(self.seed)
        rnd.shuffle(all_ids)

        num_bins = 2 ** self.B
        base = len(all_ids) // num_bins
        self.bins: List[List[int]] = []
        for i in range(num_bins):
            start = i * base
            end = (i + 1) * base if i < num_bins - 1 else len(all_ids)
            self.bins.append(all_ids[start:end])

        # 构造反向映射：token_id -> bit-block string
        self.id2bits = {}
        for i, bucket in enumerate(self.bins):
            bits = format(i, f'0{self.B}b')
            for tid in bucket:
                self.id2bits[tid] = bits

    def encrypt(self, cover: str, payload: ByteString) -> str:
        """
        忽略 cover 作为上下文前缀（或将 cover 作为初始 prompt）。
        payload: 待隐藏的二进制消息
        返回：生成的隐写文本
        """
        # 1. 转为比特串并分块
        bitstr = ''.join(f"{b:08b}" for b in payload)
        blocks = [
            bitstr[i : i + self.B]
            for i in range(0, len(bitstr), self.B)
            if len(bitstr[i : i + self.B]) == self.B
        ]

        # 2. 如果提供 cover，则将其作为初始上下文
        if cover:
            input_ids = self.tokenizer.encode(cover, return_tensors='pt')[0].tolist()
        else:
            input_ids = []

        output_ids: List[int] = []

        # 3. 逐块生成
        for block in blocks:
            bin_idx = int(block, 2)
            bin_ids = set(self.bins[bin_idx])

            # 准备模型输入
            context = torch.tensor([input_ids + output_ids], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(context)['logits'][0, -1]  # 取最后一步
            # 将不在当前桶的 token 概率置为 -inf
            mask = torch.full_like(logits, float('-inf'))
            mask[list(bin_ids)] = logits[list(bin_ids)]
            # 选择概率最大的那个 token
            next_id = torch.argmax(mask).item()

            output_ids.append(next_id)

        # 4. 输出文本
        stego = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return stego

    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        """
        根据共享分桶键，逐 token 恢复 bit-block，拼回原消息 bytes
        """
        # 1. tokenize
        token_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)

        # 2. 提取每个 token 的比特块
        bits_list: List[str] = []
        for tid in token_ids:
            bits = self.id2bits.get(tid)
            if bits is None:
                # 遇到不在任何桶中的 token，忽略
                continue
            bits_list.append(bits)

        # 3. 拼接并转 bytes
        bitstr = ''.join(bits_list)
        # 丢弃末尾不足 B 位的部分
        valid_len = (len(bitstr) // 8) * 8
        bitstr = bitstr[:valid_len]
        if not bitstr:
            return b''

        num_bytes = len(bitstr) // 8
        val = int(bitstr, 2)
        return val.to_bytes(num_bytes, byteorder='big', signed=False)
