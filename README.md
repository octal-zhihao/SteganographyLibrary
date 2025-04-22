# StegoBench
生成式文本隐写基准算法库

---

## 🧱 目录结构：StegoBench（多模型 + 多算法支持）

```
StegoBench/
├── stego_algorithms/                     # 各种隐写方法模块
│   └── bins_lstm/
│       ├── stego_engine.py              # 嵌入/解码逻辑
│       ├── vocab_bins.py                # 分 bin 工具
├── language_models/                     # 模型适配层（基于 ModelScope）
│   ├── base.py                          # BaseLanguageModel 抽象类
│   ├── gpt2_wrapper.py                  # GPT2 封装（ModelScope）
│   ├── qwen_wrapper.py                  # Qwen 封装（支持 AutoTokenizer）
├── utils/
│   ├── bit_utils.py                     # 字符串/比特转换
│   └── registry.py                      # 模型注册器
├── config/
│   └── model_config.yaml                # 配置文件（路径/模型名）
├── main.py                              # 主程序入口
└── test_cases/
    └── test_bins_lstm.py               # 示例测试脚本
```

---

## 🧠 language_models/base.py

```python
# language_models/base.py
import torch
from abc import ABC, abstractmethod

class BaseLanguageModel(ABC):
    @abstractmethod
    def encode(self, text: str) -> list:
        pass

    @abstractmethod
    def decode(self, token_ids: list) -> str:
        pass

    @abstractmethod
    def predict_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """返回最后一个 token 的 logits"""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass
```

---

## 🧠 language_models/gpt2_wrapper.py（ModelScope GPT2）

```python
# language_models/gpt2_wrapper.py
import torch
import torch.nn as nn
from modelscope import GPT2Tokenizer, GPT2Model
from language_models.base import BaseLanguageModel

class GPT2Wrapper(BaseLanguageModel):
    def __init__(self, model_path='openai-community/gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2Model.from_pretrained(model_path)
        self.proj = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.proj.to(self.device)

    def encode(self, text):
        return self.tokenizer(text)["input_ids"]

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def predict_logits(self, input_ids: torch.Tensor):
        with torch.no_grad():
            output = self.model(input_ids.to(self.device))
            hidden = output.last_hidden_state[:, -1]
            logits = self.proj(hidden).squeeze(0)
        return logits

    def get_vocab_size(self):
        return self.model.config.vocab_size
```

---

## 🧠 utils/bit_utils.py

```python
def str_to_bits(s):
    return ''.join(f'{ord(c):08b}' for c in s)

def bits_to_str(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)
```

---

## 🧠 utils/registry.py（模型工厂）

```python
# utils/registry.py
from language_models.gpt2_wrapper import GPT2Wrapper
from language_models.qwen_wrapper import QwenWrapper

MODEL_REGISTRY = {
    'gpt2': GPT2Wrapper,
    'qwen': QwenWrapper,
}

def get_model(model_name):
    return MODEL_REGISTRY[model_name]()
```

---

## 🧠 stego_algorithms/bins_lstm/stego_engine.py（简化接口）

```python
import torch
import random

def embed_bits(lm, tokenizer, bins, bit_string, max_len=32, temperature=1.0, top_k=10):
    tokens = []
    input_ids = torch.tensor([lm.encode(" ")], dtype=torch.long).to(lm.device)

    for i in range(0, len(bit_string), 2):
        bit_block = bit_string[i:i+2]
        bin_tokens = bins[bit_block]

        logits = lm.predict_logits(input_ids) / temperature
        probs = torch.softmax(logits, dim=-1)

        filtered = [(i, probs[i].item()) for i in bin_tokens if i < probs.size(0)]
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:top_k]
        ids, values = zip(*filtered)
        dist = torch.distributions.Categorical(probs=torch.tensor(values))
        next_token = ids[dist.sample().item()]

        tokens.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(lm.device)], dim=1)
        if len(tokens) >= max_len:
            break

    return tokenizer.decode(tokens)

def decode_bits(text, bins, tokenizer):
    ids = tokenizer(text)["input_ids"]
    inverse_map = {tok_id: bits for bits, tok_set in bins.items() for tok_id in tok_set}
    decoded = [inverse_map[i] for i in ids if i in inverse_map]
    return ''.join(decoded)
```

---

## 🧪 main.py 示例

```python
# main.py
import argparse
from utils.registry import get_model
from stego_algorithms.bins_lstm.vocab_bins import build_bins
from stego_algorithms.bins_lstm.stego_engine import embed_bits, decode_bits
from utils.bit_utils import str_to_bits, bits_to_str

def main(args):
    model = get_model(args.model)
    vocab_ids = list(range(model.get_vocab_size()))
    bins = build_bins(vocab_ids, num_bins=4)

    bit_string = str_to_bits(args.text)
    stego = embed_bits(model, model.tokenizer, bins, bit_string)
    print(f"[Stego Text]: {stego}")

    decoded_bits = decode_bits(stego, bins, model.tokenizer)
    print(f"[Recovered]: {bits_to_str(decoded_bits[:len(bit_string)])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2', help='gpt2 / qwen / chatglm')
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    main(args)
```

---

## ✅ 运行示例

```bash
# 切换目录到项目根目录
cd StegoBench

# 嵌入 hello 用 gpt2
python main.py --model gpt2 --text hello
```

---

## ✅ 下一步可选扩展

| 功能 | 说明 |
|------|------|
| todo 加入 qwen_wrapper.py | 支持 qwen/Qwen-7B-Chat（通过 `AutoModelForCausalLM`） |
| todo 支持多个算法 | `--algo bins_lstm` 参数切换不同算法 |
| todo 支持多评测指标 | BLEU、困惑度、GPT评分、可解性等 |

---