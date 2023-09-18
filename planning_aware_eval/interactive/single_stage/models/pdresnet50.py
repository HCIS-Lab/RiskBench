from ast import Delete
from hashlib import new
from operator import ne
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .partialconv2d import PartialConv2d
from torchvision.models.resnet import model_urls as _model_urls

__all__ = ['PDResNet', 'pdresnet18', 'pdresnet34', 'pdresnet50', 'pdresnet101', 'pdresnet152']


model_urls = {
    'pdresnet18': '',
    'pdresnet34': '',
    'pdresnet50': '/home/william/risk-assessment-via-GAT/models/model_best.pth',
    'pdresnet101': '',
    'pdresnet152': '',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, m=None):
        residual = x

        out = self.conv1(x, m)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, m)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, m


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = PartialConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PartialConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PartialConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        ###################################################
        if downsample is not None:
            self.downsample = True
            self.conv4 = PartialConv2d(inplanes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False)
            self.bn4 = nn.BatchNorm2d(planes * self.expansion)
        ###################################################

        self.stride = stride


    def forward(self, *input):
        x, m = input[0]
        residual = x

        out, m = self.conv1(x, m)
        out = self.bn1(out)
        out = self.relu(out)

        out, m = self.conv2(out, m)
        out = self.bn2(out)
        out = self.relu(out)

        out, m = self.conv3(out, m)
        out = self.bn3(out)


        if self.downsample is not None:
            residual, _ = self.conv4(x)
            residual = self.bn4(residual)
            # residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, m


class PDResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(PDResNet, self).__init__()
        self.conv1 = PartialConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     PartialConv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    #     return x

    def features(self, x, m=None):

        x, m = self.conv1(x, m)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        m = self.maxpool(m)

        x, m = self.layer1((x, m))
        x, m = self.layer2((x, m))
        x, m = self.layer3((x, m))
        x, m = self.layer4((x, m))

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x    # B*2048*10*10



def pdresnet18(pretrained=False, **kwargs):
    """Constructs a PDResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet18']))
    return model


def pdresnet34(pretrained=False, **kwargs):
    """Constructs a PDResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet34']))
    return model


def pdresnet50(pretrained=False, **kwargs):
    """Constructs a PDResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        model = PDResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model = torch.load(model_urlsss['pdresnet50'])
        return model
    """

    model = PDResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    weight = "models/backbone.pth"

    if pretrained:
        state_dict_old = torch.load(weight)
        state_dict_new = {}
        for w in state_dict_old.keys():
            state_dict_new[w[9:]] = state_dict_old[w]

        model.load_state_dict(state_dict_new)

    return model


def pdresnet101(pretrained=False, **kwargs):
    """Constructs a PDResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet101']))
    return model


def pdresnet152(pretrained=False, **kwargs):
    """Constructs a PDResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet152']))
    return model