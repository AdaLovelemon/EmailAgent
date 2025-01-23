import torch
import torch.nn as nn

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        """
        Inception Block V1
            :param in_channels: int, input channels
            :param out_channels: int, output channels
            :param num_kernels: int, number of kernels, default 6
            :param init_weight: bool, initialize weight or
            not, default True

        Inputs
        ------
            x: tensor, input tensor, (B, C_in, H, W)   

        Returns
        -------
            tensor, output tensor, (B, C_out, H, W), H and W are the same as input tensor
        
        Principles:
        -----------
            1. Use multiple kernels to extract features
            2. Stack the output of each kernel and take the average
            
        """
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    

class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        """
        Inception Block V2
            :param in_channels: int, input channels
            :param out_channels: int, output channels
            :param num_kernels: int, number of kernels, default 6
            :param init_weight: bool, initialize weight or
            not, default True

        Inputs
        ------
            x: tensor, input tensor, (B, C_in, H, W)

        Returns
        -------
            tensor, output tensor, (B, C_out, H, W), H and W are the same as input tensor

        Principles:
        -----------
            1. Use multiple kernels to extract features
            2. Stack the output of each kernel and take the average
        """
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    


if __name__ == '__main__':
    InConv = Inception_Block_V1(3, 64, 1)
    x = torch.randn(1, 3, 224, 224)
    out = InConv(x)
    print(out.shape)
