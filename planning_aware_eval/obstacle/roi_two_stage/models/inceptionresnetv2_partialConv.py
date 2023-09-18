import torch
import torch.nn as nn
from .partialconv2d import PartialConv2d
import torch.utils.model_zoo as model_zoo
import os
import sys

__all__ = ['InceptionResNetV2_Partial']

pretrained_settings = {
    'inceptionresnetv2': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}

class mySequential(nn.Sequential):
    def forward(self, x, m):
        for module in self._modules.values():
            x, m = module(x, m)

        return x, m


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = PartialConv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False, multi_channel=True, return_mask=True) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x , m):
        x, m = self.conv(x, m)
        x = self.bn(x)
        x = self.relu(x)
        return x, m


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = mySequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = mySequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        #self.branch3 = nn.Sequential(
            #nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            #BasicConv2d(192, 64, kernel_size=1, stride=1))

        self.branch3_avgpool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.branch3_conv2d = BasicConv2d(192, 64, kernel_size=1, stride=1)

        '''        nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )'''


    def forward(self, x, m):
        x0, m0 = self.branch0(x, m)
        x1, m1 = self.branch1(x, m)
        x2, m2 = self.branch2(x, m)
        x3 = self.branch3_avgpool(x)
        m3 = self.branch3_avgpool(m)
        x3, m3 = self.branch3_conv2d(x3, m3)
        out = torch.cat((x0, x1, x2, x3), 1)
        out_m = torch.cat((m0, m1, m2, m3), 1)
        return out, out_m


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = mySequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = mySequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = PartialConv2d(128, 320, kernel_size=1, stride=1 , multi_channel=True, return_mask=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x , m ):
        x0, m0 = self.branch0(x , m)
        x1, m1 = self.branch1(x, m)
        x2, m2 = self.branch2(x, m)
        out = torch.cat((x0, x1, x2), 1)
        out_m = torch.cat((m0, m1, m2), 1)
        out, out_m = self.conv2d(out, out_m)
        out = out * self.scale + x
        out = self.relu(out)
        out_m = out_m * self.scale + m
        out_m = self.relu(out_m)
        return out, out_m


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = mySequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x, m):
        x0, m0 = self.branch0(x, m)
        x1, m1 = self.branch1(x, m)
        x2 = self.branch2(x)
        m2 = self.branch2(m)
        out = torch.cat((x0, x1, x2), 1)
        out_m = torch.cat((m0, m1, m2), 1)

        return out, out_m


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = mySequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = PartialConv2d(384, 1088, kernel_size=1, stride=1 ,multi_channel=True,  return_mask=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x , m):
        x0, m0 = self.branch0(x, m)
        x1, m1 = self.branch1(x, m)
        out = torch.cat((x0, x1), 1)
        out_m = torch.cat((m0, m1), 1)

        out, out_m = self.conv2d(out, out_m)
        out = out * self.scale + x
        out = self.relu(out)
        out_m = out_m * self.scale + m
        out_m = self.relu(out_m)
        return out, out_m


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = mySequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = mySequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = mySequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x, m):
        x0, m0 = self.branch0(x, m)
        x1, m1 = self.branch1(x, m)
        x2, m2 = self.branch2(x, m)
        x3 = self.branch3(x)
        m3 = self.branch3(m)
        out = torch.cat((x0, x1, x2, x3), 1)
        out_m = torch.cat((m0, m1, m2, m3), 1)

        return out, out_m


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 =mySequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = PartialConv2d(448, 2080, kernel_size=1, stride=1, multi_channel=True, return_mask=True)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x, m):
        x0, m0 = self.branch0(x, m)
        x1, m1 = self.branch1(x, m)
        out = torch.cat((x0, x1), 1)
        out_m = torch.cat((m0, m1), 1)
        out, out_m = self.conv2d(out, out_m)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        out_m = out_m * self.scale + m

        if not self.noReLU:
            out_m = self.relu(out_m)
        return out, out_m


class InceptionResNetV2_Partial(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2_Partial, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = mySequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = mySequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = mySequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input, mask):

        x, m = self.conv2d_1a(input, mask)   # 32*149*149        
        x, m = self.conv2d_2a(x, m)       # 32*147*147
        x, m = self.conv2d_2b(x, m)       # 64*147*147
        x = self.maxpool_3a(x)      # 64*73*73
        m = self.maxpool_3a(m)      # 64*73*73
        x, m = self.conv2d_3b(x, m)       # 80*73*73
        x, m = self.conv2d_4a(x, m)       # 192*71*71
        x = self.maxpool_5a(x)      # 192*35*35
        m = self.maxpool_5a(m)      # 192*35*35
        x, m = self.mixed_5b(x, m)        # 320*35*35
        x, m = self.repeat(x, m)          # 320*35*35
        x, m = self.mixed_6a(x, m)        # 1088*17*17
        x, m = self.repeat_1(x, m)        # 1088*17*17
        x, m = self.mixed_7a(x, m)        # 2080*8*8
        x, m = self.repeat_2(x, m)        # 2080*8*8
        x, m = self.block8(x, m)          # 2080*8*8
        x, m = self.conv2d_7b(x, m)       # 1536*8*8
        return x

    '''
    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
    '''


