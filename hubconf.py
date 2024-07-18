dependencies = ["torch"]
import torch
from bigvgan_mirror import BigVGAN

URLS = {
    "bigvgan_base_22khz_80band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_base_22khz_80band.pt",
    "bigvgan_22khz_80band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_22khz_80band.pt",
    "bigvgan_base_24khz_100band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_base_24khz_100band.pt",
    "bigvgan_24khz_100band": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v1.0/bigvgan_24khz_100band.pt",
    "bigvgan_v2_44khz_128band_512x": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v2.0/bigvgan_v2_44khz_128band_512x.pt",
    "bigvgan_v2_44khz_128band_256x": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v2.0/bigvgan_v2_44khz_128band_256x.pt",
    "bigvgan_v2_24khz_100band_256x": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v2.0/bigvgan_v2_24khz_100band_256x.pt",
    "bigvgan_v2_22khz_80band_256x": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v2.0/bigvgan_v2_22khz_80band_256x.pt",
    "bigvgan_v2_22khz_80band_fmax8k_256x": "https://github.com/lars76/bigvgan-mirror/releases/download/weights-v2.0/bigvgan_v2_22khz_80band_fmax8k_256x.pt",
}


def create_bigvgan_model(config, pretrained=True, progress=True):
    model = BigVGAN(**config["model_params"])

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=URLS[config["model_name"]], map_location="cpu", progress=progress
        )
        model.load_state_dict(state_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


MODEL_CONFIGS = {
    "bigvgan_v2_44khz_128band_512x": {
        "model_name": "bigvgan_v2_44khz_128band_512x",
        "model_params": {
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "mels": 128,
            "add_bias": False,
            "add_tanh": False,
            "n_fft": 2048,
            "hop_size": 512,
            "win_size": 2048,
            "sampling_rate": 44100,
            "fmin": 0,
            "fmax": 22050,
        },
    },
    "bigvgan_v2_44khz_128band_256x": {
        "model_name": "bigvgan_v2_44khz_128band_256x",
        "model_params": {
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "mels": 128,
            "add_bias": False,
            "add_tanh": False,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 44100,
            "fmin": 0,
            "fmax": 22050,
        },
    },
    "bigvgan_v2_24khz_100band_256x": {
        "model_name": "bigvgan_v2_24khz_100band_256x",
        "model_params": {
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "mels": 100,
            "add_bias": False,
            "add_tanh": False,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 24000,
            "fmin": 0,
            "fmax": 12000,
        },
    },
    "bigvgan_v2_22khz_80band_256x": {
        "model_name": "bigvgan_v2_22khz_80band_256x",
        "model_params": {
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "mels": 80,
            "add_bias": False,
            "add_tanh": False,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 11025,
        },
    },
    "bigvgan_v2_22khz_80band_fmax8k_256x": {
        "model_name": "bigvgan_v2_22khz_80band_fmax8k_256x",
        "model_params": {
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "mels": 80,
            "add_bias": False,
            "add_tanh": False,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 8000,
        },
    },
    "bigvgan_base_22khz_80band": {
        "model_name": "bigvgan_base_22khz_80band",
        "model_params": {
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "mels": 80,
            "add_bias": True,
            "add_tanh": True,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 8000,
        },
    },
    "bigvgan_22khz_80band": {
        "model_name": "bigvgan_22khz_80band",
        "model_params": {
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "mels": 80,
            "add_bias": True,
            "add_tanh": True,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 8000,
        },
    },
    "bigvgan_base_24khz_100band": {
        "model_name": "bigvgan_base_24khz_100band",
        "model_params": {
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "mels": 100,
            "add_bias": True,
            "add_tanh": True,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 24000,
            "fmin": 0,
            "fmax": 12000,
        },
    },
    "bigvgan_24khz_100band": {
        "model_name": "bigvgan_24khz_100band",
        "model_params": {
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "mels": 100,
            "add_bias": True,
            "add_tanh": True,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 24000,
            "fmin": 0,
            "fmax": 12000,
        },
    },
}

# Create individual functions for each model
for model_name in MODEL_CONFIGS:
    exec(f"""
def {model_name}(progress: bool = True, pretrained: bool = True):
    return create_bigvgan_model(MODEL_CONFIGS['{model_name}'], pretrained, progress)
    """)
