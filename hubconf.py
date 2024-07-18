dependencies = ["torch"]

import torch
from bigvgan_mirror import BigVGAN

URLS = {
    "bigvgan_base_22khz_80band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_base_22khz_80band.pt",
    "bigvgan_22khz_80band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_22khz_80band.pt",
    "bigvgan_base_24khz_100band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_base_24khz_100band.pt",
    "bigvgan_24khz_100band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_24khz_100band.pt",
}

def bigvgan_base_22khz_80band(progress: bool = True, pretrained: bool = True):
    model = BigVGAN(upsample_rates=[8,8,2,2],
                    upsample_initial_channel=512,
                    upsample_kernel_sizes=[16,16,4,4],
                    mels=80)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["bigvgan_base_22khz_80band"], map_location="cpu", progress=progress)
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model

def bigvgan_22khz_80band(progress: bool = True, pretrained: bool = True):
    model = BigVGAN(upsample_rates=[4,4,2,2,2,2],
                    upsample_initial_channel=1536,
                    upsample_kernel_sizes=[8,8,4,4,4,4],
                    mels=80)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["bigvgan_22khz_80band"], map_location="cpu", progress=progress)
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model

def bigvgan_base_24khz_100band(progress: bool = True, pretrained: bool = True):
    model = BigVGAN(upsample_rates=[8,8,2,2],
                    upsample_initial_channel=512,
                    upsample_kernel_sizes=[16,16,4,4],
                    mels=100)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["bigvgan_base_24khz_100band"], map_location="cpu", progress=progress)
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model

def bigvgan_24khz_100band(progress: bool = True, pretrained: bool = True):
    model = BigVGAN(upsample_rates=[4,4,2,2,2,2],
                    upsample_initial_channel=1536,
                    upsample_kernel_sizes=[8,8,4,4,4,4],
                    mels=100)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["bigvgan_24khz_100band"], map_location="cpu", progress=progress)
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model