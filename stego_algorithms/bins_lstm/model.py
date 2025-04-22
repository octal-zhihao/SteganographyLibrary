# bins_lstm/model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from modelscope import GPT2Model

class StegoLSTM(pl.LightningModule):
    def __init__(self, model_name='openai-community/gpt2', lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.gpt = GPT2Model.from_pretrained(model_name)
        self.embed_proj = nn.Linear(self.gpt.config.hidden_size, self.gpt.config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.embed_proj(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)),
                                labels[:, 1:].reshape(-1))
        return loss, logits
