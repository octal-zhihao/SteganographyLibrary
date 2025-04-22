# bins_lstm/test_stego_ms.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import torch
from modelscope import GPT2Tokenizer
from bins_lstm.model import StegoLSTM
from bins_lstm.vocab_bins import build_bins
from bins_lstm.stego_engine import embed_bits, decode_bits

def str_to_bits(s):
    return ''.join(f'{ord(c):08b}' for c in s)

def bits_to_str(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

if __name__ == "__main__":
    # === 设置 ===
    model_name = 'openai-community/gpt2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    secret_text = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
    num_bits_per_block = 2

    # === 加载模型和 tokenizer ===
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = StegoLSTM(model_name=model_name).eval().to(device)

    # === 构建 bin ===
    vocab_ids = list(tokenizer.get_vocab().values())
    vocab_size = model.gpt.config.vocab_size
    bins = build_bins(vocab_ids, 2**num_bits_per_block, max_vocab_size=vocab_size)

    # === 加密 ===
    bit_string = str_to_bits(secret_text)
    print(f"[Secret Text] {secret_text}")
    print(f"[Bit String]  {bit_string}")

    # === 嵌入 bit 串，生成隐写文本 ===
    stego_text = embed_bits(model, tokenizer, bins, bit_string)
    print(f"[Stego Text]  {stego_text}")

    # === 解码回 bit 串并还原 ===
    decoded_bits = decode_bits(stego_text, bins, tokenizer)
    recovered_text = bits_to_str(decoded_bits[:len(bit_string)])
    print(f"[Recovered Bits] {decoded_bits}")
    print(f"[Recovered Text] {recovered_text}")
