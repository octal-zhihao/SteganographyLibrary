# train.py
from transformers import AutoTokenizer
from datasets import load_dataset
from pytorch_lightning import Trainer
from bins_lstm.model import StegoLSTM
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def collate_fn(batch):
    encodings = tokenizer([b['text'] for b in batch], return_tensors="pt", padding=True, truncation=True)
    return (encodings['input_ids'], encodings['attention_mask'], encodings['input_ids'])

if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    model = StegoLSTM()
    trainer = Trainer(max_epochs=3)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    trainer.fit(model, loader)
