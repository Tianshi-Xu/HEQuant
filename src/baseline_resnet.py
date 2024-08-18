import torch
import torch.nn as nn


try:
    import sys
    sys.path.append("/home/mengli/third-party/pytorch-image-models")

    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model


__all__ = ["cifar10_baseline_resnet18", "cifar10_baseline_resnet34"]


"""
In this file, we define the cifar10_baseline_resnet18 for cifar10/cifar100 dataset.
Compared to the cifar10_resnet18 in resnet.py, the major differences are:
1) no BN layers
2) use relu as activation function
3) can turn off residual by option
"""


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    # use_relu = True

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        use_relu=True,
        skip_last_relu=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = (
            nn.ReLU(inplace=True) if use_relu else nn.PReLU(planes)
        )
        self.conv2 = conv3x3(planes, planes)
        if skip_last_relu:
            self.relu2 = nn.Identity()
        else:
            self.relu2 = (
                nn.ReLU(inplace=True) if use_relu else nn.PReLU(planes)
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = x
        out = self.conv1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu1(out)
        out = self.conv2(out)
        out += identity
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        use_relu=True,
        skip_last_relu=False,
        down_block_type="default",
        **kwargs,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        if num_classes == 10 or num_classes == 100:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # END

        # self.bn1 = norm_layer(self.inplanes)
        self.bn1 = nn.Identity()
        self.relu = (
            nn.ReLU(inplace=True) if use_relu else nn.PReLU(self.inplanes)
        )
        self.layer1 = self._make_layer(
            block, 64, layers[0], use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
            use_relu=False, skip_last_relu=False, down_block_type="default"):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_block = {
                "default": [conv1x1(self.inplanes, planes * block.expansion, stride)],
                "avgpool": [
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                ],
                "conv3x3": [conv3x3(self.inplanes, planes * block.expansion, stride)],
            }[down_block_type]
            downsample = nn.Sequential(*down_block)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                use_relu=use_relu,
                skip_last_relu=skip_last_relu,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_relu=use_relu,
                    skip_last_relu=skip_last_relu,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _baseline_resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/pretrained/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


@register_model
def cifar10_baseline_resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _baseline_resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )


@register_model
def cifar10_baseline_resnet34(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _baseline_resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )
