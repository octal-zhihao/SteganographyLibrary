# language_models/base.py
import torch
from abc import ABC, abstractmethod

class BaseLanguageModel(ABC):
    @abstractmethod
    def encode(self, text: str) -> list: pass

    @abstractmethod
    def decode(self, token_ids: list) -> str: pass

    @abstractmethod
    def predict_logits(self, input_ids: torch.Tensor) -> torch.Tensor: pass

    @abstractmethod
    def get_vocab_size(self) -> int: pass
