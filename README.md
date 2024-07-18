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

wav, sr = librosa.load('/path/to/your/audio.wav', sr=22050, mono=True)
mel = torch.FloatTensor(mel_spectrogram(wav, model.n_fft, model.num_mels, model.sampling_rate, model.hop_size, model.win_size, model.fmin, model.fmax)).unsqueeze(0)

with torch.inference_mode():
    predicted_wav = model(mel) # 1 x T tensor (16-bit integer)
```

## Benchmark

You can run `python benchmark.py` to compare the performance between the original code and this one.

## Available models

| Model                        | Sampling Rate | Mel Band | fmax  | Params | Dataset                       |
|------------------------------|---------------|----------|-------|--------|-------------------------------|
| bigvgan_24khz_100band        | 24 kHz        | 100      | 12000 | 112M   | LibriTTS                      |
| bigvgan_base_24khz_100band   | 24 kHz        | 100      | 12000 | 14M    | LibriTTS                      |
| bigvgan_22khz_80band         | 22 kHz        | 80       | 8000  | 112M   | LibriTTS + VCTK + LJSpeech    |
| bigvgan_base_22khz_80band    | 22 kHz        | 80       | 8000  | 14M    | LibriTTS + VCTK + LJSpeech    |