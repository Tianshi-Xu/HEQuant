import pdb
import torch.nn as nn
# from torchsummary import summary
from timm.models.registry import register_model
from src.mobilenetv2_tiny import mobilenet_tiny
__all__ = ["mobilev2"]

def conv_dw(ch_in, stride=1):
    return nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False)
            

def conv_pw(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False)


def conv3x3(ch_in, ch_out, stride=1):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False)

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride, use_bn, use_relu):
        super(InvertedBlock, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_bn = use_bn
        self.use_relu = use_relu

        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        if expand_ratio == 1: #t=1的block
            self.conv_dw = conv_dw(hidden_dim, stride=stride)
            self.bn1 = norm_layer(hidden_dim) if use_bn else nn.Identity()
            self.relu1 = (
                nn.ReLU6(inplace=True) if use_relu else nn.PReLU(hidden_dim)
            )

            self.conv_pwl = conv_pw(hidden_dim, ch_out)
            self.bn2 = norm_layer(ch_out) if use_bn else nn.Identity()
        else:
            self.conv_pw = conv_pw(ch_in, hidden_dim)
            self.bn1 = norm_layer(hidden_dim) if use_bn else nn.Identity()
            self.relu1 = (
                nn.ReLU6(inplace=True) if use_relu else nn.PReLU(hidden_dim)
            )

            self.conv_dw = conv_dw(hidden_dim, stride=stride)
            self.bn2 = norm_layer(hidden_dim) if use_bn else nn.Identity()
            self.relu2 = (
                nn.ReLU6(inplace=True) if use_relu else nn.PReLU(hidden_dim)
            )

            self.conv_pwl = conv_pw(hidden_dim, ch_out)
            self.bn3 = norm_layer(ch_out) if use_bn else nn.Identity()

      

    def forward(self, x):
        identity = x
        if self.expand_ratio == 1:

            out = self.conv_dw(x)
            out = self.bn1(out)
            out = self.relu1(out)

            out = self.conv_pwl(out)
            out = self.bn2(out)

        else:
            
            out = self.conv_pw(x)   
            out = self.bn1(out)
            out = self.relu1(out)

            out = self.conv_dw(out)
            out = self.bn2(out)
            out = self.relu2(out)

            out = self.conv_pwl(out)
            out = self.bn3(out)

        if self.use_res_connect:
            return out + identity
        else:
            return out

class MobileNetV2(nn.Module):
    def __init__(
        self, 
        ch_in=3, 
        n_classes=1000, 
        use_bn=True,
        use_relu=True,
        **kargs
        ):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.conv_stem = conv3x3(ch_in, 32, stride=2)

        
        input_channel = 32
        self.block = InvertedBlock
        self.blocks = nn.ModuleList([])
        for t,c,n,s in self.configs:
            self.blocks.append(self._make_stage(inchannel=input_channel, outchannel=c, expantion=t, repeat=n, stride=s,\
                                use_bn=use_bn, use_relu=use_relu))
            input_channel = c

        # layers = []
        # for t, c, n, s in self.configs:
        #     for i in range(n):
        #         stride = s if i == 0 else 1
        #         layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride,\
        #                        use_bn=use_bn, use_relu=use_relu))
        #         input_channel = c
        # self.layers = nn.Sequential(*layers)

        self.conv_head = conv_pw(input_channel, 1280)
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = (
                nn.ReLU6(inplace=True) if use_relu else nn.PReLU(hidden_dim)
            )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(1280, n_classes)
        )

    def _make_stage(self,expantion,inchannel,outchannel,repeat,stride,use_bn,use_relu):
        strides = [stride]+[1]*repeat 
        # pdb.set_trace()
        blocks = []
        for i in range(repeat):
            blocks.append(self.block(inchannel,outchannel,expand_ratio=expantion,stride= strides[i],\
                         use_bn=use_bn, use_relu=use_relu))
            inchannel = outchannel
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv_stem(x)
        for b in self.blocks:
            x = b(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x


@register_model
def mobilev2(pretrained=False, progress=False, **kwargs):
    return MobileNetV2(ch_in=3, n_classes=1000, **kwargs)
    
@register_model
def tinyimagenet_mobilenetv2(pretrained=False, **kwargs):
    model=mobilenet_tiny(200,64,1)
    return model
