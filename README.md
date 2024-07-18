# bigvgan-mirror

A mirror of BigVGAN for access via PyTorch Hub. There are no dependencies other than PyTorch. I cleaned up the original code from [NVIDIA](https://github.com/NVIDIA/BigVGAN.git). The weights here are for prediction. If you want to train BigVGAN, you also need the discriminator and have to add weight_norm to the generator.

## Example Usage

Below is an example demonstrating how to generate a mel spectrogram from an audio file and use BigVGAN to synthesize audio from it.

```python
import torch
import librosa
import numpy as np

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    # Create mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )

    # Pad the signal
    pad_length = int((n_fft - hop_size) / 2)
    y = np.pad(y, (pad_length, pad_length), mode="reflect")

    # Compute STFT
    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=False,
        pad_mode="reflect",
    )

    # Convert to magnitude spectrogram and add small epsilon
    S = np.sqrt(np.abs(D) ** 2 + 1e-9)

    # Apply mel filterbank
    S = np.dot(mel_basis, S)

    # Convert to log scale
    S = np.log(np.maximum(S, 1e-5))

    return S

model = torch.hub.load("lars76/bigvgan-mirror", "bigvgan_base_22khz_80band",
                       trust_repo=True, pretrained=True)

wav, sr = librosa.load('/path/to/your/audio.wav', sr=model.sampling_rate, mono=True)
mel = torch.FloatTensor(mel_spectrogram(wav, model.n_fft, model.num_mels, model.sampling_rate, model.hop_size, model.win_size, model.fmin, model.fmax)).unsqueeze(0)

with torch.inference_mode():
    predicted_wav = model(mel) # 1 x T tensor (16-bit integer)
```

## Benchmark

You can run `python benchmark.py` to compare the performance between the original code and this one.

This table presents the results of evaluating the PESQ (Perceptual Evaluation of Speech Quality) score on 20 WAV files from the AISHELL-3 dataset. The evaluation compares the performance of the cleaned up model with the original model.

| Model Name                            | Our Model PESQ ± StdDev  | Original Model PESQ ± StdDev |
|---------------------------------------|--------------------------|------------------------------|
| bigvgan_base_22khz_80band             | 3.5663 ± 0.2969          | 3.5709 ± 0.3007              |
| bigvgan_22khz_80band                  | 3.9896 ± 0.2263          | 3.9903 ± 0.2270              |
| bigvgan_base_24khz_100band            | 3.6254 ± 0.3467          | 3.6300 ± 0.3398              |
| bigvgan_24khz_100band                 | 4.0109 ± 0.2970          | 4.0123 ± 0.2957              |
| bigvgan_v2_44khz_128band_512x         | 3.7666 ± 0.3043          | 3.7670 ± 0.3044              |
| bigvgan_v2_44khz_128band_256x         | 3.8612 ± 0.4234          | 3.8614 ± 0.4240              |
| bigvgan_v2_24khz_100band_256x         | **4.1534 ± 0.3198**      | **4.1535 ± 0.3204**          |
| bigvgan_v2_22khz_80band_256x          | 3.9525 ± 0.2713          | 3.9525 ± 0.2709              |
| bigvgan_v2_22khz_80band_fmax8k_256x   | 4.0168 ± 0.2664          | 4.0165 ± 0.2682              |

The difference is <0.01, which proves that the code is correct.

## Model descriptions

| Model                         | Sampling Rate | Mel Band | fmax  | Params | Dataset                       |
|-------------------------------|---------------|----------|-------|--------|-------------------------------|
| bigvgan_24khz_100band         | 24 kHz        | 100      | 12000 | 112M   | LibriTTS                      |
| bigvgan_base_24khz_100band    | 24 kHz        | 100      | 12000 | 14M    | LibriTTS                      |
| bigvgan_22khz_80band          | 22 kHz        | 80       | 8000  | 112M   | LibriTTS + VCTK + LJSpeech    |
| bigvgan_base_22khz_80band     | 22 kHz        | 80       | 8000  | 14M    | LibriTTS + VCTK + LJSpeech    |
| bigvgan_v2_44khz_128band_512x | 44 kHz        | 128      | 22050 | 122M   | Large-scale Compilation       |
| bigvgan_v2_44khz_128band_256x | 44 kHz        | 128      | 22050 | 112M   | Large-scale Compilation       |
| bigvgan_v2_24khz_100band_256x | 24 kHz        | 100      | 12000 | 112M   | Large-scale Compilation       |
| bigvgan_v2_22khz_80band_256x  | 22 kHz        | 80       | 11025 | 112M   | Large-scale Compilation       |