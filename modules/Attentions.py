import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "embed_size must be divisible by num_heads"
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x, mask):
        N, seq_length, embed_size = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, value)
        out = out.transpose(1, 2).contiguous().view(N, seq_length, embed_size)
        out = self.fc_out(out)
        return out