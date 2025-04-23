from language_models.gpt2_wrapper import GPT2Wrapper
from language_models.qwen_wrapper import QwenWrapper

MODEL_REGISTRY = {
    'gpt2': GPT2Wrapper,
    'qwen': QwenWrapper,
}

def get_model(model_name: str):
    return MODEL_REGISTRY[model_name]()
