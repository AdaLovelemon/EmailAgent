import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        """
        Inputs:
        ------
            x: [B, C, T]
        """
        super(ConvAutoEncoder, self).__init__()

        # Encoder: Convolutional layers to learn feature representations
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # (batch, input_channels, seq_len)
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Decoder: Convolutional layers to reconstruct the input signal
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # For normalized output (between 0 and 1)
        )

    def forward(self, x):
        # Encoder pass
        encoded = self.encoder(x)
        
        # Decoder pass
        decoded = self.decoder(encoded)
        
        return decoded


