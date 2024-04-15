
import torch
from stage2.models.wrappers import ResidualWrapper, UpsampleConvWrapper
from huggingface_hub import PyTorchModelHubMixin

class ImageTransfomer(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
    
        self.relu_activation = torch.nn.ReLU()

        # Down-sampling convolution layers
        num_of_channels = [3, 32, 64, 128]
        kernel_sizes = [9, 3, 3]
        stride_sizes = [1, 2, 2]
        self.conv1 = torch.nn.Conv2d(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0], stride=stride_sizes[0], padding=kernel_sizes[0]//2, padding_mode='reflect')
        self.in1 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.conv2 = torch.nn.Conv2d(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1], stride=stride_sizes[1], padding=kernel_sizes[1]//2, padding_mode='reflect')
        self.in2 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.conv3 = torch.nn.Conv2d(num_of_channels[2], num_of_channels[3], kernel_size=kernel_sizes[2], stride=stride_sizes[2], padding=kernel_sizes[2]//2, padding_mode='reflect')
        self.in3 = torch.nn.InstanceNorm2d(num_of_channels[3], affine=True)

        # Residual layers
        res_block_num_of_filters = 128
        self.residual1 = ResidualWrapper(res_block_num_of_filters)
        self.residual2 = ResidualWrapper(res_block_num_of_filters)
        self.residual3 = ResidualWrapper(res_block_num_of_filters)
        self.residual4 = ResidualWrapper(res_block_num_of_filters)
        self.residual5 = ResidualWrapper(res_block_num_of_filters)

        # Up-sampling convolution layers
        num_of_channels.reverse()
        kernel_sizes.reverse()
        stride_sizes.reverse()
        self.up1 = UpsampleConvWrapper(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0], stride=stride_sizes[0])
        self.in4 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.up2 = UpsampleConvWrapper(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1], stride=stride_sizes[1])
        self.in5 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.up3 = torch.nn.Conv2d(num_of_channels[2], num_of_channels[3], kernel_size=kernel_sizes[2], stride=stride_sizes[2], padding=kernel_sizes[2]//2, padding_mode='reflect')

    def forward(self, x):
        y = self.relu_activation(self.in1(self.conv1(x)))
        y = self.relu_activation(self.in2(self.conv2(y)))
        y = self.relu_activation(self.in3(self.conv3(y)))
        y = self.residual1(y)
        y = self.residual2(y)
        y = self.residual3(y)
        y = self.residual4(y)
        y = self.residual5(y)
        y = self.relu_activation(self.in4(self.up1(y)))
        y = self.relu_activation(self.in5(self.up2(y)))
        
        return self.up3(y)

