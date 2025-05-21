import abc
import math
from typing import ByteString, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers import (
    BertConfig,
    RobertaConfig,
    DistilBertConfig,
    GPT2Config,
    BloomConfig,
    OPTConfig,
)

class StegoMethod(abc.ABC):
    @abc.abstractmethod
    def encrypt(self, cover: str, payload: ByteString) -> str:
        """将 payload 嵌入 cover 文本，返回 stego 文本"""
        pass

    @abc.abstractmethod
    def decrypt(self, cover: str, stego_text: str) -> ByteString:
        """给定原始 cover 和 stego_text，提取并返回 payload"""
        pass

# 基础加载器
def load_masked_lm(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    if not isinstance(config, (BertConfig, RobertaConfig, DistilBertConfig)):
        raise ValueError(
            f"模型 `{model_name}` 不是一个 Masked Language Model。"
            " 请使用支持 Masked LM 的模型，例如 `bert-base-uncased`、`roberta-base` 等。"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

def load_causal_lm(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    if not isinstance(config, (GPT2Config, BloomConfig, OPTConfig)):
        raise ValueError(
            f"模型 `{model_name}` 不是一个常见的因果语言模型 (Causal LM)。"
            " 请使用 GPT-2 / Bloom / OPT 等支持 CausalLM 的模型名称，"
            "例如 `gpt2`, `gpt2-medium`, `bloom-560m`, `facebook/opt-350m`。"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def limit_past(past_key_values, max_length: Optional[int] = None):
    """
    Truncate GPT-2 style past_key_values so that past_len <= max_length.
    If max_length is None, infer from model config (default to 1023).
    """
    if past_key_values is None:
        return None

    # 如果外部没提供 max_length，就取一个安全值 1023
    max_len = max_length or 1023

    truncated = []
    for key, value in past_key_values:
        # key/value shape: (batch, n_head, seq_len, head_dim)
        seq_len = key.size(-2)
        if seq_len > max_len:
            key   = key[..., -max_len:, :].contiguous()
            value = value[..., -max_len:, :].contiguous()
        truncated.append((key, value))
    return tuple(truncated)

def kl(q, logq, logp):
    res = q*(logq-logp)/0.69315
    res[q==0] = 0
    return res.sum().item() # in bits

def entropy(q, logq):
    res = q*logq/0.69315
    res[q==0] = 0
    return -res.sum().item() # in bits

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return '.' in token or '!' in token or '?' in token

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i
