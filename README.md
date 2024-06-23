# bigvgan-mirror

A mirror of BigVGAN for access via PyTorch Hub. There are no dependencies other than PyTorch. I cleaned up the original code from [NVIDIA](https://github.com/NVIDIA/BigVGAN.git). The weights here are for prediction. If you want to train BigVGAN, you also need the discriminator and have to add weight_norm to the generator.

```python
import torch

model = torch.hub.load("lars76/bigvgan-mirror", "bigvgan_base_22khz_80band",
					   trust_repo=True, pretrained=True)
mel_spectogram = torch.randn(1, 80, 40)
predicted_wav = model(mel_spectogram) # 1 x 10240
```

## Available models

|Folder Name|Sampling Rate|Mel band|fmax|Params.|Dataset|Fine-Tuned|
|------|---|---|---|---|------|---|
|bigvgan_24khz_100band|24 kHz|100|12000|112M|LibriTTS|No|
|bigvgan_base_24khz_100band|24 kHz|100|12000|14M|LibriTTS|No|
|bigvgan_22khz_80band|22 kHz|80|8000|112M|LibriTTS + VCTK + LJSpeech|No|
|bigvgan_base_22khz_80band|22 kHz|80|8000|14M|LibriTTS + VCTK + LJSpeech|No|