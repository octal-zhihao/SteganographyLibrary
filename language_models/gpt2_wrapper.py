# language_models/gpt2_wrapper.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from language_models.base import BaseLanguageModel

class GPT2Wrapper(BaseLanguageModel):
    def __init__(self, model_path='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt')[0].tolist()

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def predict_logits(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device))
            return outputs.logits[0, -1]  # shape: [vocab_size]

    def get_vocab_size(self):
        return self.model.config.vocab_size
