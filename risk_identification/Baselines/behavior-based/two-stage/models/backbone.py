import torch
import torch.nn as nn
import timm
from torchvision.ops import roi_align


class ROI_ALIGN(nn.Module):
    def __init__(self, kernel_size, n, scale=1.0):
        """
            kernel_size: roi align kernel
            n: number of objects
        """
        super().__init__()
        self.roi_align = roi_align
        self.kernel = kernel_size
        self.scale = scale
        self.global_img = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        # self.conv1 = nn.Conv2d(2048, 512, 1)
        # self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.n = n

    def forward(self, features, boxes):
        b = len(boxes)
        boxes = list(boxes)

        # crops = torchvision.ops.roi_align(input_image, [torch.FloatTensor(
        # [[60, 80, 160, 220], [250, 250, 500, 500]]).to(device)], output_size=(200, 300))

        x = self.roi_align(features, boxes, [self.kernel, self.kernel], self.scale)
        x = self.global_img(x)
        x = x.reshape(b, self.n, -1)
        
        return x


class Riskbench_backbone(nn.Module):
    def __init__(self, roi_align_kernel, n, pretrained=True, backbone='resnet50'):
        """
            backbone: specify which backbone ['resnet50', 'resnet101', ...]
        """
        super(Riskbench_backbone, self).__init__()
        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, out_indices=[-1])
        self.object_layer = ROI_ALIGN(roi_align_kernel, n, 1./32.)

        self.set_finetune()

    def forward(self, img, bbox=None):
        """
            box : List[Tensor[N,4]] # box: (b t) n 4
        """
        object = None
        img = self.backbone(img)
        if bbox is not None:
            object = self.object_layer(img[0], bbox)  # (b t) N 2048 8 8

        return img, object

    def set_finetune(self, tune_list=['layer4']):
        """
            backbone structure:
                layer1 layer2 layer3 layer4
        """
        for para in self.backbone.parameters():
            para.requires_grad = False
        for name, para in self.backbone.named_parameters():
            layer = name.split('.')[0]
            if layer in tune_list:
                para.requires_grad = True



if __name__ == '__main__':

    tmp_model = Riskbench_backbone(8, 20)
    tmp = torch.rand((10, 3, 256, 640))
    bbox = torch.rand((10, 20, 4))
    bbox = list(bbox)

    # bbox = torch.Tensor(bbox)
    tmp, object = tmp_model(tmp, bbox)
