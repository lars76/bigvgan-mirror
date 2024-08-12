import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding="same",
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="same"),
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="same"),
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="same"),
            ]
        )

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.lrelu(x)
            xt = c1(xt)
            xt = self.lrelu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
            ]
        )
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        for c in self.convs:
            xt = self.lrelu(x)
            xt = c(xt)
            x = xt + x
        return x


class HifiGAN(nn.Module):
    def __init__(
        self,
        resblock_type,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
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
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.fmin = fmin
        self.fmax = fmax

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_mels = mels

        self.conv_pre = Conv1d(mels, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock_type == 1 else ResBlock2

        self.ups = nn.ModuleList(
            [
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
            ]
        )

        self.resblocks = nn.ModuleList(
            [
                resblock(upsample_initial_channel // (2 ** (i + 1)), k, d)
                for i in range(self.num_upsamples)
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = Conv1d(
            upsample_initial_channel // (2**self.num_upsamples), 1, 7, 1, padding=3
        )
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.lrelu(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.lrelu(x)
        x = self.conv_post(x)
        return self.tanh(x).squeeze(1)
