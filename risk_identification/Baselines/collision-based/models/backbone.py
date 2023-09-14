import torch
import torch.nn as nn
import timm
from torchvision.ops import roi_align

class ROI_ALIGN(nn.Module):
    def __init__(self,kernel_size,scale=1.0):
        """
            kernel_size: roi align kernel
            n: number of objects
        """
        super().__init__()
        self.roi_align = roi_align
        self.kernel = kernel_size
        self.scale = scale
        self.object_projection = nn.Sequential(
            nn.Conv2d(2048,512,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self,features, boxes):
        b = len(boxes)
        x = self.roi_align(features, boxes, self.kernel, self.scale, aligned = False)
        x = self.object_projection(x)
        return x

class Riskbench_backbone(nn.Module):
    def __init__(self,roi_align_kernel,n,backbone='resnet50',intention=False):
        """
            backbone: specify which backbone ['resnet50', 'resnet101', ...]
            n: number of objects
        """
        super(Riskbench_backbone, self).__init__()
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=True,out_indices=[-1])
        self.object_layer = ROI_ALIGN(roi_align_kernel,1.0/32.0)
        self.n = n
        dim = 128 if intention else 256
        self.global_img = nn.Sequential(
            nn.Conv2d(2048,512,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,dim,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.set_finetune()

    def forward(self, img, bbox=None):
        """
            box : List[Tensor[N,4]] # box: (b t) n 4
        """
        tensor_dim = img.ndim
        if tensor_dim == 5:
            b, t, c, H, W = img.shape
            img = img.reshape(b*t,c,H,W)
            if isinstance(bbox,torch.Tensor):
                bbox = bbox.reshape(b*t,-1,4)
                bbox = list(bbox)
        elif tensor_dim == 4:
            b = img.shape[0]
            if isinstance(bbox,torch.Tensor):
                bbox = list(bbox)

        object = None
        img = self.backbone(img)[0]
        if bbox is not None:
            object = self.object_layer(img,bbox) 
        img = self.global_img(img)
        if tensor_dim == 5:
            object = object.reshape(b,t,self.n,-1)
            img = img.reshape(b,t,-1)
        elif tensor_dim == 4:
            object = object.reshape(b,self.n,-1)
            img = img.reshape(b,-1)
        return img, object
    
    def set_finetune(self,tune_list=['layer4']):
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