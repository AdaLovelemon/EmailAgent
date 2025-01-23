import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, d_in, d_out, num_out):
        """
        Args
        ----
            d_in: Last layers' number of features's dimension 
            d_out: Expected output vector's number of dimension
            num_out: Possibility number of each dimension of output vectors 
        """
        super().__init__()
        self.d_out = d_out
        self.num_out = num_out
        self.linear = nn.Linear(d_in, d_out * num_out)
        
    def forward(self, x:torch.Tensor):
        batch_size, _ = x.shape
        x = self.linear(x)
        x = x.reshape(batch_size, self.d_out, self.num_out)
        return x.softmax(dim=-1)