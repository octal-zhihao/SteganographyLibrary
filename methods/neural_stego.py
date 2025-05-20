from utils import StegoMethod, load_causal_lm
class NeuralStego(StegoMethod):
    def __init__(self, model_name='gpt2'):
        self.tokenizer, self.model = load_causal_lm(model_name)
    def encrypt(self, cover: str, payload: bytes) -> str:
        # TODO: 算术编码实现
        raise NotImplementedError
    def decrypt(self, stego_text: str) -> bytes:
        raise NotImplementedError