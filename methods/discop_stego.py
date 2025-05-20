from utils import StegoMethod, load_causal_lm
class DiscopStego(StegoMethod):
    def __init__(self, model_name='gpt2'):
        self.tokenizer, self.model = load_causal_lm(model_name)
    def encrypt(self, cover: str, payload: bytes) -> str:
        # TODO: 分布副本实现
        raise NotImplementedError
    def decrypt(self, stego_text: str) -> bytes:
        raise NotImplementedError