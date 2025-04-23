# language_models/qwen_wrapper.py
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
from language_models.base import BaseLanguageModel

class QwenWrapper(BaseLanguageModel):
    def __init__(self, model_path='qwen/Qwen-7B-Chat'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def encode(self, text):
        # 注意：Qwen tokenizer 的 encode 返回 dict，需获取 input_ids
        return self.tokenizer(text, return_tensors='pt')['input_ids'][0].tolist()

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def predict_logits(self, input_ids: torch.Tensor):
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1]  # 最后一个 token 的分布
        return logits

    def get_vocab_size(self):
        return self.model.config.vocab_size
