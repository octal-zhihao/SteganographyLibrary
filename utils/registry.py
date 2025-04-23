from language_models.gpt2_wrapper import GPT2Wrapper
from language_models.qwen_wrapper import QwenWrapper
from language_models.opt_wrapper import OPTWrapper
MODEL_REGISTRY = {
    'gpt2': GPT2Wrapper,
    'qwen': QwenWrapper,
    'opt': OPTWrapper,
}

def get_model(model_name: str):
    return MODEL_REGISTRY[model_name]()
