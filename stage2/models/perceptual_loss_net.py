from collections import namedtuple
import torch
from torchvision import models
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        vgg_pretrained = models.vgg16(pretrained=True, progress=show_progress).features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'] 
        
        self.conv1_1 = vgg_pretrained[0]
        self.relu1_1 = vgg_pretrained[1]
        self.conv1_2 = vgg_pretrained[2]
        self.relu1_2 = vgg_pretrained[3]
        self.max_pooling1 = vgg_pretrained[4]
        self.conv2_1 = vgg_pretrained[5]
        self.relu2_1 = vgg_pretrained[6]
        self.conv2_2 = vgg_pretrained[7]
        self.relu2_2 = vgg_pretrained[8]
        self.max_pooling2 = vgg_pretrained[9]
        self.conv3_1 = vgg_pretrained[10]
        self.relu3_1 = vgg_pretrained[11]
        self.conv3_2 = vgg_pretrained[12]
        self.relu3_2 = vgg_pretrained[13]
        self.conv3_3 = vgg_pretrained[14]
        self.relu3_3 = vgg_pretrained[15]
        self.max_pooling3 = vgg_pretrained[16]
        self.conv4_1 = vgg_pretrained[17]
        self.relu4_1 = vgg_pretrained[18]
        self.conv4_2 = vgg_pretrained[19]
        self.relu4_2 = vgg_pretrained[20]
        self.conv4_3 = vgg_pretrained[21]
        self.relu4_3 = vgg_pretrained[22]
        self.max_pooling4 = vgg_pretrained[23]
        self.conv5_1 = vgg_pretrained[24]
        self.relu5_1 = vgg_pretrained[25]
        self.conv5_2 = vgg_pretrained[26]
        self.relu5_2 = vgg_pretrained[27]
        self.conv5_3 = vgg_pretrained[28]
        self.relu5_3 = vgg_pretrained[29]
        self.max_pooling5 = vgg_pretrained[30]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1_1(x)
        conv1_1 = x
        x = self.relu1_1(x)
        relu1_1 = x
        x = self.conv1_2(x)
        conv1_2 = x
        x = self.relu1_2(x)
        relu1_2 = x
        x = self.max_pooling1(x)
        x = self.conv2_1(x)
        conv2_1 = x
        x = self.relu2_1(x)
        relu2_1 = x
        x = self.conv2_2(x)
        conv2_2 = x
        x = self.relu2_2(x)
        relu2_2 = x
        x = self.max_pooling2(x)
        x = self.conv3_1(x)
        conv3_1 = x
        x = self.relu3_1(x)
        relu3_1 = x
        x = self.conv3_2(x)
        conv3_2 = x
        x = self.relu3_2(x)
        relu3_2 = x
        x = self.conv3_3(x)
        conv3_3 = x
        x = self.relu3_3(x)
        relu3_3 = x
        x = self.max_pooling3(x)
        x = self.conv4_1(x)
        conv4_1 = x
        x = self.relu4_1(x)
        relu4_1 = x
        x = self.conv4_2(x)
        conv4_2 = x
        x = self.relu4_2(x)
        relu4_2 = x
        x = self.conv4_3(x)
        conv4_3 = x
        x = self.relu4_3(x)
        relu4_3 = x
        x = self.max_pooling4(x)
        x = self.conv5_1(x)
        conv5_1 = x
        x = self.relu5_1(x)
        relu5_1 = x
        x = self.conv5_2(x)
        conv5_2 = x
        x = self.relu5_2(x)
        relu5_2 = x
        x = self.conv5_3(x)
        conv5_3 = x
        x = self.relu5_3(x)
        relu5_3 = x
        x = self.max_pooling5(x)
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out
    
PerceptualLossNet = Vgg16
