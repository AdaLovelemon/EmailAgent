import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
            Embedding Module
            ================
            This module is used to embed the input tokens to the desired dimension,
            which is utilized to map the input tokens to the higher dimensional continuous space.
            Usually for **vocabulary set**.

            This is a simple embedding module assembled the nn.Embedding module.

            parameters:
            -----------
            num_embeddings: int
                The number of unique tokens in the input tensor.
            embedding_dim: int
                The dimension of the output embedding.

            Inputs:
            -------
            input: torch.Tensor
                The input tensor of shape (B, T)
            
            Outputs:
            --------
            x: torch.Tensor
                The output tensor of shape (B, T, embedding_dim)

            Principals:
            -----------
            Usually, for a given vocabulary set, a computer cannot grasp the meanings of each word in the vocabulary set.
            What a computer can excel at is to compute the similarities between the words in the vocabulary set. In that cases,
            mapping discrete indices which correpond to the words in the vocabulary set to the continuous space is a good idea.
            You can just select the dimension of the output embedding, which is the dimension of the continuous space.
            
            Given a specific vocabulary index, the module would lookup the corresponding embedding vector in the weight matrix
            just like a dictionary. The weight matrix is initialized with uniform distribution and updated during the training process.
            And the optimization goal for embeddings may differ according to the task, e.g., classification, regression, etc.
            Whatever, the embedding module is a bridge between the discrete space and the continuous space, making the discrete words a
            more tackleable form for the computer.
        """
        super(Embedding, self).__init__()
        # Initialize the weight matrix
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        # Reset the parameters with uniform distribution
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight)

    def forward(self, input):
        # Use the F.embedding function to embed the input tensor
        return F.embedding(input, self.weight)

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False):
    """
        Embedding Function
        ==================
        This function is used to embed the input tensor to the desired dimension, 
        a simple example that assembles the F.embedding function.
    """
    if padding_idx is not None:
        weight = weight.index_fill(0, torch.tensor([padding_idx]), 0)
    if max_norm is not None:
        input = input.contiguous()
        with torch.no_grad():
            norms = weight.norm(norm_type, 1, keepdim=True)
            desired = torch.clamp(norms, max=max_norm)
            weight = weight * (desired / norms)
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
            Positional Embedding Module
            ===========================
            This module is used to add positional information to the input tensor.
            The positional encodings have the same dimension as the input tensor.

            parameters:
            -----------
            d_model: int
                The dimension of the input tensor.
            max_len: int
                The maximum length of the input tensor.

            Inputs:
            -------
            x: torch.Tensor
                The input tensor of shape (B, T, d_model)
            (B: Batch Size, T: Sequence Length, d_model: Dimension of the input tensor)
            Outputs:
            --------
            pe: torch.Tensor
                The output tensor of shape (B, T, d_model)

            Principals:
            -----------
            The positional encodings are computed using the following formulas:
            Given i is the dimension index, d_model is the dimension of the input tensor,
            and pos is the position of the token in the input tensor.
            
            PE(pos, 2i) = sin(pos / 10000^(2i / d_model)) 

            PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # Positional Indeces, (max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        # Scaling Factor, (d_model // 2,) 
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term) # Even Indeces
        pe[:, 1::2] = torch.cos(position * div_term) # Odd Indeces

        # Add a batch dimension, (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # Register pe in the buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Return the positional encodings for the input tensor, same time steps
        return self.pe[:, :x.size(1)]
    

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
            Token Embedding Module
            ======================
            This module is used to embed the input tokens to the desired dimension,
            which is utilized to capture the local patterns in the input tokens.

            Usually used to tokenize **time series data**.

            parameters:
            -----------
            c_in: int
                The dimension of the input tokens.
            d_model: int
                The dimension of the output embedding.

            Inputs:
            -------
            x: torch.Tensor
                The input tensor of shape (B, T, c_in)
            (B: Batch Size, T: Sequence Length, c_in: Dimension of the input tokens)
            
            Outputs:
            --------
            x: torch.Tensor
                The output tensor of shape (B, T, d_model)

            Principals:
            -----------
            The token embeddings are computed using the following formulas:
            Given x is the input tensor, c_in is the dimension of the input tokens,
            and d_model is the dimension of the output embedding.

            x = Conv1d(x, d_model, kernel_size=3, padding=1, padding_mode='circular')
            
            However, input tensor's shape should be permuted before applying the convolution,
            i.e., x = x.permute(0, 2, 1)
            
            1D Convolution helps to extract **local patterns** in the input tokens.
            And PyTorch's Conv1d needs the input tensor to be in the shape of (B, C, T),
            which means fusing a patch of time steps into a single token.
            
        """
        super(TokenEmbedding, self).__init__()
        # Equal length convolution, `circular` padding ensures information always from these time steps
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        Fixed Embedding Module
        ======================
        This module assembles `PositionalEmbedding` Module to compute positional encodings.
        However, its inputs are discrete indices, not the continuous input tensor,
        and the positional encodings are fixed, not size-varied.
        
        Usually used to embed **time series data**.

        parameters:
        -----------
        c_in: int
            The number of unique tokens in the input tensor.
        d_model: int
            The dimension of the output embedding.
        
        Inputs:
        -------
        x: torch.Tensor
            The input tensor of shape (B, T)
        (B: Batch Size, T: Sequence Length)

        Outputs:
        --------
        x: torch.Tensor
            The output tensor of shape (B, T, d_model)
        
        """
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # nn.Embedding module uses a weight matrix to embed the input tokens, which shape is (c_in, d_model)
        self.emb = nn.Embedding(c_in, d_model)
        # Reset the embedding weights as the fixed positional encodings
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        Temporal Embedding Module
        =========================
        This module is used to embed the temporal information (e.g. hours, dates, months) in the input tensor.
        Usually used to embed **time series data**.

        parameters:
        -----------
        d_model: int
            The dimension of the output embedding.
        embed_type: str
            The type of the embedding module, `fixed` or `learned`.
        freq: str
            The frequency of the time series data, `h` for hourly, `t` for minutely.

        Inputs:
        -------
        x: torch.Tensor
            The input tensor of shape (B, T, num_features)
            (B: Batch Size, T: Sequence Length, num_features: Number of features),
            where num_feaures is 5 when it is (month, day, weekday, hour, minute) or 4 when it is (month, day, weekday, hour).
        
        Outputs:
        --------    
        x: torch.Tensor
            The output tensor of shape (B, T, d_model)

        Principals:
        -----------
        The temporal features (e.g. hours, week_days, dates, months, etc.) are discrete, they need to be embedded to the higher dimensional continuous space
        in order to better capture the temporal patterns in the time series data.

        Each temporal has its own embedding module, either fixed (`FixedEmbedding`) or learnable (`nn.Embedding`), and the output embeddings are summed up to represent the temporal information in the input tensor.
        
        """
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        Time Feature Embedding Module
        =============================
        This module is used to embed the temporal information (e.g. hours, dates, months) in the input tensor.
        Different from `TemporalEmbedding`, this module embeds the temporal information using a linear layer.

        parameters:
        -----------
        d_model: int
            The dimension of the output embedding.
        embed_type: str
            The type of the embedding module, `timeF` for time feature embedding.
        freq: str
            The frequency of the time series data, `h` for hourly, `t` for minutely.

        Inputs:
        -------
        x: torch.Tensor
            The input tensor of shape (B, T, num_features)
            (B: Batch Size, T: Sequence Length, num_features: Number of features),
            where num_feaures varies according to the following dictionary:

            freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
            

        Outputs:
        --------
        x: torch.Tensor
            The output tensor of shape (B, T, d_model)

        """
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq] # Select the input dimension according to the frequency

        # Embed the temporal features using a linear layer
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Data Embedding Module
        =====================
        This module is used to combine Value Embedding `TokenEmbedding`, Positional Embedding `PositionalEmbedding`, Temporal Embedding `TemporalEmbedding` or `TimeFeatureEmbedding` 
        to embed the input tensors.

        parameters:
        -----------
        c_in: int
            The dimension of the input tokens.
        d_model: int
            The dimension of the output embedding.
        embed_type: str
            The type of the embedding module, 'fixed' or 'timeF'. If is 'timeF', the `TimeFeatureEmbedding` is used.
        freq: str
            The frequency of the time series data, 'h' for hourly, 't' for minutely.
        dropout: float
            The dropout rate.

        Inputs:
        -------
        x: torch.Tensor
            The input tensor of shape (B, T, c_in)
            (B: Batch Size, T: Sequence Length, c_in: Dimension of the input tokens)
        x_mark: torch.Tensor
            The input tensor contains information about temporal features (e.g. months, dates, hours, etc.) of shape (B, T, num_features)
            (B: Batch Size, T: Sequence Length, num_features: Number of features). 
            Please select correct `num_features` according to the frequency. 

        Outputs:
        --------
        x: torch.Tensor
            The output tensor of shape (B, T, d_model)
        
        """
        
        super(DataEmbedding, self).__init__()

        # Value Embedding
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # Positional Embedding
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # Temporal Embedding
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # If x_mark is None, use the value embedding and the positional embedding
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        
        # Else, use the value embedding, the temporal embedding, and the positional embedding
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Data Embedding Module without Positional Embedding
        ==================================================
        This module is used to combine Value Embedding `TokenEmbedding`, Temporal Embedding `TemporalEmbedding` or `TimeFeatureEmbedding`
        to embed the input tensors without the positional encodings.

        parameters:
        -----------
        c_in: int
            The dimension of the input tokens.
        d_model: int
            The dimension of the output embedding.
        embed_type: str
            The type of the embedding module, 'fixed' or 'timeF'. If is 'timeF', the `TimeFeatureEmbedding` is used.
        freq: str
            The frequency of the time series data, 'h' for hourly, 't' for minutely.
        dropout: float
            The dropout rate.

        Inputs:
        -------
        x: torch.Tensor
            The input tensor of shape (B, T, c_in)
            (B: Batch Size, T: Sequence Length, c_in: Dimension of the input tokens)
        x_mark: torch.Tensor
            The input tensor contains information about temporal features (e.g. months, dates, hours, etc.) of shape (B, T, num_features)
            (B: Batch Size, T: Sequence Length, num_features: Number of features). 
            Please select correct `num_features` according to the frequency. 

        Outputs:
        --------
        x: torch.Tensor
            The output tensor of shape (B, T, d_model)
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
    

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        """
        Data Embedding Module with Inverted Dimensions
        ========================================================
        This module is used to combine only one kind of embeddings, the Value Embedding `TokenEmbedding`, and Temporal Features
        to embed the input tensors. The input tensors are permuted to match the expected input shape for the linear layer.

        parameters:
        -----------
        c_in: int
            The dimension of the input tensors plus temporal tensors' dimension.
        d_model: int
            The dimension of the output embedding.
        embed_type: str
            The type of the embedding module, 'fixed' or 'timeF'. If 'timeF', the `TimeFeatureEmbedding` is used.
        freq: str
            The frequency of the time series data, 'h' for hourly, 't' for minutely.
        dropout: float
            The dropout rate.

        Inputs:
        -------
        x: torch.Tensor
            The input tensor of shape (B, T, c_in)
            (B: Batch Size, T: Sequence Length, c_in: Dimension of the input tokens)
        x_mark: torch.Tensor
            The input tensor contains information about temporal features (e.g. months, dates, hours, etc.) of shape (B, T, num_features)
            (B: Batch Size, T: Sequence Length, num_features: Number of features). 
            Please select correct `num_features` according to the frequency. 

        Outputs:
        --------
        x: torch.Tensor
            The output tensor of shape (B, Variate, d_model)

        """
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [B, T, c_in]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            print(x.size(1), x_mark.size(1))
            print(torch.cat([x, x_mark], 2).shape)
            x = self.value_embedding(torch.cat([x, x_mark], 2))
        # x: [B, T, d_model]
        return self.dropout(x)
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        """
        Patch Embedding Module
        ======================
        This module is used to patch the input tensor to the desired dimension.
        Usually used to patch the input tensor for **Time Series data** as this one is for 1D data. 

        parameters:
        -----------
        d_model: int
            The dimension of the output embedding.
        patch_len: int
            The length of the patch.
        stride: int
            The stride of the patching.
        padding: int
            The padding of the patching.
        dropout: float
            The dropout rate.

        Inputs:
        ------- 
        x: torch.Tensor
            The input tensor of shape (B, C, H, W)
            (B: Batch Size, C: Number of Channels, H: Height, W: Width)

        Outputs:
        --------
        x: torch.Tensor
            The output tensor of shape (B, T, d_model)
            (B: Batch Size, T: Sequence Length, d_model: Dimension of the output embedding)

        """
        
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        # Padding the input tensor
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars



class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout_rate=0.1):
        """
        Bert Embedding Module
        =====================
        This module is used to embed the input tokens to the desired dimension,
        which is utilized to map the input tokens to the higher dimensional continuous space.

        Usually used to embed the input tokens for **NLP tasks**.

        parameters:
        -----------
        vocab_size: int
            The size of vocabulary set, usually the size of vocabulary set that used in the pre-trained model.
            For example, the size of the vocabulary set in the BERT-base model is 30522.
        embed_size: int
            The dimension of the output embedding. In BERT-base, the dimension of the output embedding is 768.
        max_len: int
            The maximum length of the input tensor. In BERT-base, the dimension of the input tensor is 512.
        dropout_rate: float
            The dropout rate.
        
        Inputs:
        -------
        input_ids: torch.Tensor
            The input tensor of shape (B, T), which symbolizes the input word by its index in the vocabulary set.
            (B: Batch Size, T: Sequence Length)
        segment_ids: torch.Tensor
            The input tensor of shape (B, T)
            (B: Batch Size, T: Sequence Length)
        
        Outputs:
        --------
        embeddings: torch.Tensor
            The output tensor of shape (B, T, embed_size)
            (B: Batch Size, T: Sequence Length, embed_size: Dimension of the output embedding)

        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size) # Word Embedding, transform the input word by indices to the vector
        self.position_embedding = nn.Embedding(max_len, embed_size) # Positional Embedding, add positional information to the input tensor
        self.segment_embedding = nn.Embedding(2, embed_size) # Segment Embedding, used to distinguish between two sentences
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.token_embedding(input_ids) # (B, seq_len, embed_size)
        position_embeddings = self.position_embedding(position_ids) # (B, seq_len, embed_size)
        segment_embeddings = self.segment_embedding(segment_ids) # (B, seq_len, embed_size)

        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
