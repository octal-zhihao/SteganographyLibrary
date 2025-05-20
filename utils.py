import abc
import math
from typing import ByteString
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
    def decrypt(self, stego_text: str) -> ByteString:
        """从 stego 文本中提取原始 payload"""
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