# bins_lstm/stego_engine.py
import torch
import random

def embed_bits(model, tokenizer, bins, bit_string, max_len=32, temperature=1.0, top_k=10):
    tokens = []
    input_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(model.device)

    for i in range(0, len(bit_string), 2):
        bit_block = bit_string[i:i+2]
        bin_tokens = bins[bit_block]

        outputs = model.gpt(input_ids)
        last_hidden = outputs.last_hidden_state[:, -1]  # shape: [1, hidden]
        logits = model.embed_proj(last_hidden).squeeze(0) / temperature
        probs = torch.softmax(logits, dim=-1)

        # 过滤合法 token
        filtered = [(i, probs[i].item()) for i in bin_tokens if i < probs.size(0)]
        if not filtered:
            raise ValueError(f"No valid token IDs in bin {bit_block}")

        # 按概率排序，取前 top_k
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:top_k]
        ids, values = zip(*filtered)
        values = torch.tensor(values)
        dist = torch.distributions.Categorical(probs=values)
        sampled_idx = dist.sample().item()
        next_token_id = ids[sampled_idx]

        tokens.append(next_token_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(model.device)], dim=1)

        if len(tokens) >= max_len:
            break

    return tokenizer.decode(tokens)



def decode_bits(text, bins, tokenizer):
    ids = tokenizer(text)["input_ids"]
    inverse_map = {tok_id: bits for bits, tok_set in bins.items() for tok_id in tok_set}
    decoded = [inverse_map[i] for i in ids if i in inverse_map]
    return ''.join(decoded)
