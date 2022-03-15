import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch




def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out







class DeResNet(models.ResNet):  
    """docstring for DeResNet"""
    def __init__(self, block, layers, **kwargs):
        super(DeResNet, self).__init__(block, layers, **kwargs)
        self.inplanes = 512
        self.bn1 = None
        self.conv2 = None
        self.maxpool = None
        self.fc = nn.Linear(10, 512)
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.layer4 = self._make_layer(block, 64, layers[3])
        
        self.conv1 = nn.ConvTranspose2d(self.inplanes, 3, kernel_size=7, stride=1, padding=3,  bias=False)
        
        self.register_buffer('norm',torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        



    def _make_layer(self, block, planes, blocks,
                    stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if True:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, block.expansion * planes, stride),
                norm_layer(block.expansion * planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = block.expansion * planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)





    def _forward_impl(self, x):
        # See note [TorchScript super()]
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        '''
        bs = x.shape[0]
        x =  self.fc(x).view(bs, -1, 1, 1).repeat(1, 1, 7, 7)
        x = F.upsample(self.layer1(x),scale_factor=2)
        x = F.upsample(self.layer2(x),scale_factor=2)
        x = F.upsample(self.layer3(x),scale_factor=2)

        x = F.upsample(self.layer4(x),scale_factor=2)

        x = self.conv1(x)
        x = F.upsample(x,scale_factor=2)
        #x = x * self.std + self.norm

        return x

    def forward(self, x):
        return self._forward_impl(x)
		
def _resnet(
    arch,
    block,
    layers,
    **kwargs
):
    model = DeResNet(block, layers, **kwargs)
    
    return model


def deresnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],
                   **kwargs)

'''

torch.Size([32, 64, 112, 112])
torch.Size([32, 64, 56, 56])
torch.Size([32, 64, 56, 56])
torch.Size([32, 128, 28, 28])
torch.Size([32, 256, 14, 14])
torch.Size([32, 512, 7, 7])
torch.Size([32, 512])

'''