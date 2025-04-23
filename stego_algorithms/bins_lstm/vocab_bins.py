# stego_algorithms/bins_lstm/vocab_bins.py
import random

def build_bins(vocab_ids, num_bins=4, seed=42, max_vocab_size=None):
    """
    将 vocab ID 列表划分为 num_bins 个 bin，用于比特块映射。
    :param vocab_ids: list[int]，词表中的 token id
    :param num_bins: int，分成多少个 bin（例如 4 表示 2bit）
    :param seed: int，随机种子，确保 Alice 和 Bob 一致
    :param max_vocab_size: int，可选，限制最大词表 id
    :return: dict[str, set[int]]，形如 {'00': {1,2,...}, '01': {...}, ...}
    """
    random.seed(seed)
    if max_vocab_size is not None:
        vocab_ids = [v for v in vocab_ids if v < max_vocab_size]

    shuffled = vocab_ids.copy()
    random.shuffle(shuffled)

    bin_size = len(shuffled) // num_bins
    bins = {}
    for i in range(num_bins):
        bit_key = format(i, f'0{int(num_bins).bit_length() - 1}b')
        bins[bit_key] = set(shuffled[i * bin_size: (i + 1) * bin_size])

    return bins
