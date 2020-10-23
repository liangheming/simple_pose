import torch
from torch import nn
from nets import resnet
from nets.common import DUC


class FastPose(nn.Module):
    def __init__(self, num_joints=17, backbone="resnet50", norm_layer=None, reduction=True, pretrained=True):
        super(FastPose, self).__init__()
        self.backbones = getattr(resnet, backbone)(pretrained=pretrained, reduction=reduction, norm_layer=norm_layer)
        self.shuffle = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
        self.conv = nn.Conv2d(
            128, num_joints, kernel_size=3, stride=1, padding=1)
        self._initialize()

    def forward(self, x):
        x = self.backbones(x)
        x = self.shuffle(x)
        x = self.duc1(x)
        x = self.duc2(x)
        x = self.conv(x)
        return x

    def _initialize(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    input_tensor = torch.rand(size=(4, 3, 256, 192))
    net = FastPose()
    out = net(input_tensor)
    print(out.shape)
