# bigvgan-mirror

A mirror of BigVGAN for access via PyTorch Hub. There are no dependencies other than PyTorch. I cleaned up the original code from [NVIDIA](https://github.com/NVIDIA/BigVGAN.git). The weights here are for prediction. If you want to train BigVGAN, you also need the discriminator and have to add weight_norm to the generator.

## Example Usage

Below is an example demonstrating how to generate a mel spectrogram from an audio file and use BigVGAN to synthesize audio from it.

```python
import torch
import librosa
import numpy as np

def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False):
    # Create mel filterbank
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    
    # Pad the signal
    pad_length = int((n_fft - hop_size) / 2)
    y = np.pad(y, (pad_length, pad_length), mode='reflect')
    
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, 
                     window='hann', center=center, pad_mode='reflect')
    
    # Convert to magnitude spectrogram and add small epsilon
    S = np.sqrt(np.abs(D)**2 + 1e-9)
    
    # Apply mel filterbank
    S = np.dot(mel_basis, S)
    
    # Convert to log scale
    S = np.log(np.maximum(S, 1e-5))
    
    return S

wav, sr = librosa.load('/path/to/your/audio.wav', sr=22050, mono=True)
mel_spectogram = torch.FloatTensor(mel_spectrogram(wav)).unsqueeze(0)

model = torch.hub.load("lars76/bigvgan-mirror", "bigvgan_base_22khz_80band",
					   trust_repo=True, pretrained=True)
predicted_wav = model(mel_spectogram) # 1 x T tensor (16-bit integer)
```

## Available models

| Model                        | Sampling Rate | Mel Band | fmax  | Params | Dataset                       |
|------------------------------|---------------|----------|-------|--------|-------------------------------|
| bigvgan_24khz_100band        | 24 kHz        | 100      | 12000 | 112M   | LibriTTS                      |
| bigvgan_base_24khz_100band   | 24 kHz        | 100      | 12000 | 14M    | LibriTTS                      |
| bigvgan_22khz_80band         | 22 kHz        | 80       | 8000  | 112M   | LibriTTS + VCTK + LJSpeech    |
| bigvgan_base_22khz_80band    | 22 kHz        | 80       | 8000  | 14M    | LibriTTS + VCTK + LJSpeech    |