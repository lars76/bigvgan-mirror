import torch
import librosa
import numpy as np
from pesq import pesq
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(1, "BigVGAN")
import bigvgan
from meldataset import get_mel_spectrogram
from hubconf import URLS

WAV_DATASET_PATH = Path("../dataset").rglob("*.wav")
MAX_FILES = 20


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    mel_basis = librosa.filters.mel(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )
    pad_length = int((n_fft - hop_size) / 2)
    y = np.pad(y, (pad_length, pad_length), mode="reflect")
    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=False,
        pad_mode="reflect",
    )
    S = np.sqrt(np.abs(D) ** 2 + 1e-9)
    S = np.dot(mel_basis, S)
    return np.log(np.maximum(S, 1e-5))


def calculate_pesq(orig_wav, predicted_wav, sr):
    orig_wav = librosa.resample(orig_wav, orig_sr=sr, target_sr=16000)
    predicted_wav = librosa.resample(predicted_wav, orig_sr=sr, target_sr=16000)
    return pesq(16000, orig_wav, predicted_wav, "wb")


def ours(files, model_name):
    model = torch.hub.load(
        "lars76/bigvgan-mirror",
        model_name,
        source="github",
        trust_repo=True,
        pretrained=True,
    )
    pesq_scores = []
    for filename in tqdm(files, desc="Our model"):
        orig_wav, sr = librosa.load(filename, sr=model.sampling_rate, mono=True)
        mel = torch.FloatTensor(
            mel_spectrogram(
                orig_wav,
                model.n_fft,
                model.num_mels,
                model.sampling_rate,
                model.hop_size,
                model.win_size,
                model.fmin,
                model.fmax,
            )
        ).unsqueeze(0)
        with torch.inference_mode():
            predicted_wav = model(mel).squeeze(0).numpy()
        pesq_scores.append(calculate_pesq(orig_wav, predicted_wav, sr))
    return np.mean(pesq_scores), np.std(pesq_scores)


def original(files, model_name):
    model = bigvgan.BigVGAN.from_pretrained(
        f"nvidia/{model_name}", use_cuda_kernel=False
    )
    model.remove_weight_norm()
    model = model.eval()
    pesq_scores = []
    for filename in tqdm(files, desc="Original model"):
        orig_wav, sr = librosa.load(filename, sr=model.h.sampling_rate, mono=True)
        mel_spectrogram = get_mel_spectrogram(
            torch.FloatTensor(orig_wav).unsqueeze(0), model.h
        )
        with torch.inference_mode():
            predicted_wav = model(mel_spectrogram).squeeze(0).squeeze(0).numpy()
        pesq_scores.append(calculate_pesq(orig_wav, predicted_wav, sr))
    return np.mean(pesq_scores), np.std(pesq_scores)


def main():
    files = sorted(list(WAV_DATASET_PATH))[:MAX_FILES]
    for model_name in URLS.keys():
        print(f"Model name: {model_name}")
        our_mean, our_std = ours(files, model_name)
        orig_mean, orig_std = original(files, model_name)
        print(f"Ours: {our_mean:.4f} ± {our_std:.4f}")
        print(f"Original: {orig_mean:.4f} ± {orig_std:.4f}")
        print()


if __name__ == "__main__":
    main()
