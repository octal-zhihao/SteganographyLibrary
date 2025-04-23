# main.py
import argparse
from utils.registry import get_model
from stego_algorithms.bins_lstm.vocab_bins import build_bins
from stego_algorithms.bins_lstm.stego_engine import embed_bits, decode_bits
from utils.bit_utils import str_to_bits, bits_to_str

def main(args):
    model = get_model(args.model)
    vocab_ids = list(range(model.get_vocab_size()))
    bins = build_bins(vocab_ids, num_bins=4)

    bit_string = str_to_bits(args.text)
    stego = embed_bits(model, model.tokenizer, bins, bit_string)
    print(f"[Stego Text]: {stego}")

    decoded_bits = decode_bits(stego, bins, model.tokenizer)
    print(f"[Recovered]: {bits_to_str(decoded_bits[:len(bit_string)])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2', help='gpt2 / qwen')
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    main(args)
