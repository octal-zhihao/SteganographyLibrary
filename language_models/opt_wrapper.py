# language_models/opt_wrapper.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from language_models.base import BaseLanguageModel

class OPTWrapper(BaseLanguageModel):
    def __init__(self, model_path='facebook/opt-1.3b'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.device = self.model.device

    def encode(self, text: str) -> list:
        return self.tokenizer(text, return_tensors='pt').input_ids[0].tolist()

    def decode(self, token_ids: list) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def predict_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits
        return logits[0, -1, :]

    def get_vocab_size(self) -> int:
        return self.model.config.vocab_size
