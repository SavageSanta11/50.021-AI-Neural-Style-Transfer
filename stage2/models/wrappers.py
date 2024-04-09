import torch

class ResidualWrapper(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualWrapper, self).__init__()
        kernel_size = 3
        stride_size = 1
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride_size, padding=kernel_size//2, padding_mode='reflect')
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride_size, padding=kernel_size//2, padding_mode='reflect')
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual  

class UpsampleConvWrapper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.upsampling_factor = stride
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)

