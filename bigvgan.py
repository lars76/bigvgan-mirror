import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn import Conv1d, ConvTranspose1d, Parameter

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    #input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1),
                       stride=self.stride, groups=C)

        return out

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)

        return xx

class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x

class SnakeBeta(nn.Module):
    def __init__(self, in_features):
        super(SnakeBeta, self).__init__()
        self.alpha = Parameter(torch.zeros(in_features))
        self.beta = Parameter(torch.zeros(in_features))

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1)) # line up with x to [B, C, T]
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))

        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

class AMPBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(AMPBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))
        ])

        self.convs2 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)),
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)),
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))
        ])

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        self.activations = nn.ModuleList([
            Activation1d(
                activation=SnakeBeta(channels))
             for _ in range(self.num_layers)
        ])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

class BigVGAN(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, mels):
        super(BigVGAN, self).__init__()
        self.num_upsamples = len(upsample_rates)
        self.resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
        self.resblock_kernel_sizes = [3,7,11]
        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_mels = mels

        # pre conv
        self.conv_pre = Conv1d(self.num_mels, upsample_initial_channel, 7, 1, padding=3)

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                            upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2)
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(ch, k, d))

        # post conv
        activation_post = SnakeBeta(ch)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x