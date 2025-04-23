import torch

def embed_bits(lm, tokenizer, bins, bit_string, max_len=32, temperature=1.0, top_k=10):
    tokens = []
    input_ids = torch.tensor([lm.encode(" ")], dtype=torch.long).to(lm.model.device)

    for i in range(0, len(bit_string), 2):
        bit_block = bit_string[i:i+2]
        bin_tokens = bins[bit_block]

        logits = lm.predict_logits(input_ids) / temperature
        probs = torch.softmax(logits, dim=-1)
        filtered = [(i, probs[i].item()) for i in bin_tokens if i < probs.size(0)]
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:top_k]
        ids, values = zip(*filtered)
        dist = torch.distributions.Categorical(probs=torch.tensor(values))
        next_token = ids[dist.sample().item()]

        tokens.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(lm.model.device)], dim=1)
        if len(tokens) >= max_len: break

    return tokenizer.decode(tokens)

def decode_bits(text, bins, tokenizer):
    ids = tokenizer(text)["input_ids"]
    inverse_map = {tok_id: bits for bits, tok_set in bins.items() for tok_id in tok_set}
    decoded = [inverse_map[i] for i in ids if i in inverse_map]
    return ''.join(decoded)
