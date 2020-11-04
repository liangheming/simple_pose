import math
import torch
from torch import nn


def model_scale(name="s"):
    name_dict = {
        "s": (0.33, 0.50),
        "m": (0.67, 0.75),
        "l": (1.00, 1.00),
        "x": (1.33, 1.25)
    }
    multiples = name_dict.get(name, None)
    if multiples is None:
        raise NotImplementedError("scale_name only support s,m,l,x")
    return multiples


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def depth_grow(x: int, depth_multiples: float):
    return max(round(x * depth_multiples), 1) if x > 1 else x


def width_grow(x, width_multiples):
    return make_divisible(x * width_multiples, 8)


# noinspection PyUnresolvedReferences
class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, act=nn.Hardswish):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, padding=None, groups=1):
        super(Focus, self).__init__()
        self.conv = CBR(in_channel * 4, out_channel, kernel, stride, padding, groups)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        x = self.conv(x)
        return x


# noinspection PyUnresolvedReferences
class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, shortcut=True, groups=1, expansion=0.5, act=nn.Hardswish):
        super(BottleNeck, self).__init__()
        inner_channel = int(out_channel * expansion)
        self.conv1 = CBR(in_channel, inner_channel, 1, 1, act=act)
        self.conv2 = CBR(inner_channel, out_channel, 3, 1, groups=groups, act=act)
        self.add = shortcut and inner_channel == out_channel

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add:
            out = x + out
        return out


# noinspection PyUnresolvedReferences
class BottleNeckCSP(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, shortcut=True, groups=1, expansion=0.5, act=nn.Hardswish):
        super(BottleNeckCSP, self).__init__()
        inner_channel = int(out_channel * expansion)
        self.conv1_0 = CBR(in_channel, inner_channel, 1, 1, act=act)
        self.conv2_0 = nn.Conv2d(in_channel, inner_channel, 1, 1, bias=False)
        self.conv1_n = nn.Conv2d(inner_channel, inner_channel, 1, 1, bias=False)
        self.conv3 = CBR(2 * inner_channel, out_channel, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * inner_channel)
        self.act = act() if act else nn.Identity()
        self.conv1_s = nn.Sequential(*[BottleNeck(inner_channel, inner_channel, shortcut, groups, expansion=1, act=act)
                                       for _ in range(blocks)])

    def forward(self, x):
        y1 = self.conv1_n(self.conv1_s(self.conv1_0(x)))
        y2 = self.conv2_0(x)
        y = self.act(self.bn(torch.cat([y1, y2], dim=1)))
        y = self.conv3(y)
        return y


# noinspection PyUnresolvedReferences
class BottleNeckCSP2(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, shortcut=False, groups=1, expansion=0.5, act=nn.Hardswish):
        super(BottleNeckCSP2, self).__init__()
        inner_channel = int(out_channel * expansion)
        self.conv0 = CBR(in_channel, inner_channel, 1, 1, act=act)
        self.conv2 = nn.Conv2d(inner_channel, inner_channel, 1, 1, bias=False)
        self.conv3 = CBR(2 * inner_channel, out_channel, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * inner_channel)
        self.act = act() if act else nn.Identity()
        self.conv1 = nn.Sequential(
            *[BottleNeck(inner_channel, inner_channel, shortcut=shortcut, groups=groups, expansion=1.0, act=act)
              for _ in range(blocks)])

    def forward(self, x):
        x1 = self.conv0(x)
        y1 = self.conv1(x1)
        y2 = self.conv2(x1)
        y = self.conv3(self.act(self.bn(torch.cat([y1, y2], dim=1))))
        return y


# noinspection PyUnresolvedReferences
class SPP(nn.Module):
    def __init__(self, in_channel, out_channel, k=(5, 9, 13), act=nn.Hardswish):
        super(SPP, self).__init__()
        inner_channel = in_channel // 2
        self.conv1 = CBR(in_channel, inner_channel, 1, 1, act=act)
        self.conv2 = CBR(inner_channel * (len(k) + 1), out_channel, 1, 1, act=act)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat(([x] + [pool(x) for pool in self.pools]), dim=1)
        x = self.conv2(x)
        return x


# noinspection PyUnresolvedReferences
class SPPCSP(nn.Module):
    def __init__(self, in_channel, out_channel, expansion=0.5, k=(5, 9, 13), act=nn.Hardswish):
        super(SPPCSP, self).__init__()
        inner_channel = int(2 * out_channel * expansion)
        self.conv2 = nn.Conv2d(in_channel, inner_channel, 1, 1, bias=False)
        self.conv1_start = nn.Sequential(CBR(in_channel, inner_channel, 1, 1, act=act),
                                         CBR(inner_channel, inner_channel, 3, 1, act=act),
                                         CBR(inner_channel, inner_channel, 1, 1, act=act)
                                         )
        self.conv1_spp = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

        self.conv1_end = nn.Sequential(CBR(4 * inner_channel, inner_channel, 1, 1, act=act),
                                       CBR(inner_channel, inner_channel, 3, 1, act=act))
        self.bn = nn.BatchNorm2d(2 * inner_channel)
        self.act = act() if act else nn.Identity()
        self.conv3 = CBR(2 * inner_channel, out_channel, 1, 1, act=act)

    def forward(self, x):
        x1 = self.conv1_start(x)
        y1 = self.conv1_end(torch.cat([x1] + [m(x1) for m in self.conv1_spp], dim=1))
        y2 = self.conv2(x)
        y = self.conv3(self.act(self.bn(torch.cat([y1, y2], dim=1))))
        return y

# if __name__ == '__main__':
#     input_tensor = torch.rand(size=(1, 32, 128, 128))
#     net = SPPCSP(32, 64, act=nn.LeakyReLU)
#     out = net(input_tensor)
#     print(out.shape)
