# language_models/qwen_wrapper.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from language_models.base import BaseLanguageModel

class QwenWrapper(BaseLanguageModel):
    def __init__(self, model_path='Qwen/Qwen2.5-3B-Instruct'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, text):
        return self.tokenizer(text, return_tensors='pt')['input_ids'][0].tolist()

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def predict_logits(self, input_ids: torch.Tensor):
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1]
        return logits

    def get_vocab_size(self):
        return self.model.config.vocab_size
