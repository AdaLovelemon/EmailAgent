import torch
import torch.nn as nn

from modules.TransformerBlock import TransformerBlock
from modules.Embed import BertEmbedding

class BERT(nn.Module):
    def __init__(self, config):
        """
        BERT model
            :param config: dict, model configuration

        Inputs
        ------
            input_ids: tensor, input tensor, (B, L)
            segemnt_ids: tensor, segment tensor, (B, L)
            mask: tensor, mask tensor, which is used to indicate which positions are valid (1) and invalid (0) (B, L)

        Returns
        -------
            logits: tensor, output logits, (B, L, vocab_size)
        """

        vocab_size = getattr(config, 'vocab_size', 30522 )
        embed_size = getattr(config, 'embed_size', 768)
        num_heads = getattr(config, 'num_heads', 12)
        ff_hidden_size = getattr(config, 'ff_hidden_size', 3072)
        num_layers = getattr(config, 'num_layers', 12)
        max_len = getattr(config, 'max_len', 512)
        dropout_rate = getattr(config, 'dropout', 0.1)

        super().__init__()
        self.embedding = BertEmbedding(vocab_size, embed_size, max_len, dropout_rate)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout_rate) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids, segemnt_ids, mask):
        x = self.embedding(input_ids, segemnt_ids)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.fc(x)
        return logits
