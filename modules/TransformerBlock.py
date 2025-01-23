import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Attentions import MultiHeadSelfAttention

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + x)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout_rate)
        
    def forward(self, x, mask):
        attention = self.attention(x, mask)
        out = self.feed_forward(attention)
        return out