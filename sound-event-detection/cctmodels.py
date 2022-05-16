from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vit_pytorch.cct import CCT

class CCTNet(nn.Module):
    def __init__(self):
        super(CCTNet, self).__init__()
        self.cct = CCT(img_size = (256, 4800),
            embedding_dim = 384,
            n_conv_layers = 2,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 14,
            num_heads = 6,
            mlp_radio = 3.,
           n_input_channels = 4,
            num_classes = 14,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none'])
        )
        self.fc1 = nn.Linear(14, 25200)

    def forward(self, x):
        x = self.cct(x)
        x = F.relu(self.fc1(x))
        x = x.reshape(1,600,42)
        return x
