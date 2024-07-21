import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    beta = (
        0.1102 * (A - 8.7)
        if A > 50.0
        else 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
        if A >= 21.0
        else 0.0
    )
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    time = (
        torch.arange(-half_size, half_size) + 0.5
        if even
        else torch.arange(kernel_size) - half_size
    )
    filter_ = (
        torch.zeros_like(time)
        if cutoff == 0
        else 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    )
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride=1,
        padding=True,
        padding_mode="replicate",
        kernel_size=12,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x,
            self.filter.expand(x.shape[1], -1, -1),
            stride=self.stride,
            groups=x.shape[1],
        )


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
            ),
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.expand(x.shape[1], -1, -1),
            stride=self.stride,
            groups=x.shape[1],
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio=2,
        down_ratio=2,
        up_kernel_size=12,
        down_kernel_size=12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class SnakeBeta(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        alpha = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
        beta = torch.exp(self.beta).unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=d, padding="same"
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=1, padding="same"
                )
                for _ in range(3)
            ]
        )
        self.activations = nn.ModuleList(
            [Activation1d(activation=SnakeBeta(channels)) for _ in range(6)]
        )

    def forward(self, x):
        for c1, c2, a1, a2 in zip(
            self.convs1, self.convs2, self.activations[::2], self.activations[1::2]
        ):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGAN(nn.Module):
    def __init__(
        self,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        mels,
        n_fft,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        add_bias=False,
        add_tanh=False,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.fmin = fmin
        self.fmax = fmax

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = 3
        self.num_mels = mels

        self.conv_pre = nn.Conv1d(mels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose1d(
                            upsample_initial_channel // (2**i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k in [3, 7, 11]:
                self.resblocks.append(AMPBlock1(ch, k, [1, 3, 5]))

        self.activation_post = Activation1d(activation=SnakeBeta(ch))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=add_bias)
        self.add_tanh = add_tanh

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = self.ups[i][0](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.add_tanh:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)

        return x.squeeze(1)
