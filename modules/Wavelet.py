import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class WaveletTransformLayer(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        """
        Args:
            wavelet: The type of wavelet to use, e.g., 'db1' (Haar), 'db2', etc.
            level: Number of levels to decompose the signal
        """
        super(WaveletTransformLayer, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        """
        Apply Discrete Wavelet Transform to input x
        Args:
            x: Tensor with shape (batch_size, channels, time_steps)
        Returns:
            Tensor with the wavelet-transformed features
        """
        batch_size, channels, time_steps = x.size()
        # Prepare a tensor to hold the transformed data
        wavelet_features = []

        for i in range(batch_size):
            # Process each example in the batch individually
            signal = x[i].cpu().detach().numpy()  # Convert to numpy for pywt

            # Perform the wavelet transform on each channel
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

            # Combine all the coefficients (approximations + details)
            wavelet_coeffs = []
            for coeff in coeffs:
                wavelet_coeffs.append(torch.tensor(coeff, dtype=torch.float32))
            
            # Stack the wavelet coefficients along the channel dimension
            wavelet_features.append(torch.stack(wavelet_coeffs))

        # Convert list to tensor and return as the output
        return torch.stack(wavelet_features).to(x.device)
    

if __name__ == '__main__':
    model = WaveletTransformLayer()
    x = torch.randn(4, 300, 3)
    y = model(x.transpose(1, 2))
    print(y.shape)