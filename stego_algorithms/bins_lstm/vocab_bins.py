# bins_lstm/vocab_bins.py
def build_bins(vocab, num_bins, seed=42, max_vocab_size=None):
    import random
    random.seed(seed)

    if max_vocab_size is not None:
        vocab = [v for v in vocab if v < max_vocab_size]

    shuffled = vocab.copy()
    random.shuffle(shuffled)
    bins = {}
    bin_size = len(shuffled) // num_bins
    for i in range(num_bins):
        bit_key = format(i, f'0{int(num_bins).bit_length() - 1}b')
        bins[bit_key] = set(shuffled[i * bin_size: (i + 1) * bin_size])
    return bins
