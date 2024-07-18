import torch
import glob
import librosa
import numpy as np
from pesq import pesq
from tqdm import tqdm

import sys

sys.path.insert(1, "BigVGAN")
import bigvgan
from meldataset import get_mel_spectrogram

WAV_DATASET_PATH = "../dataset/**/*.wav"
MAX_FILES = 20
MODEL_NAME = "bigvgan_base_22khz_80band"


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


def ours(files):
    model = torch.hub.load(
        "lars76/bigvgan-mirror",
        MODEL_NAME,
        source="github",
        trust_repo=True,
        pretrained=True,
    )
    pesq_score = []
    for filename in tqdm(files):
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

        predicted_wav = librosa.resample(
            predicted_wav / 32767, orig_sr=sr, target_sr=16000
        )
        orig_wav = librosa.resample(orig_wav, orig_sr=sr, target_sr=16000)
        pesq_score.append(pesq(16000, orig_wav, predicted_wav, "wb"))

    print(f"Ours: {np.mean(pesq_score)} +- {np.std(pesq_score)}")


def original(files):
    model = bigvgan.BigVGAN.from_pretrained(
        f"nvidia/{MODEL_NAME}", use_cuda_kernel=False
    )
    model.remove_weight_norm()
    model = model.eval()

    pesq_score = []
    for filename in tqdm(files):
        orig_wav, sr = librosa.load(filename, sr=22050, mono=True)
        mel_spectogram = get_mel_spectrogram(
            torch.FloatTensor(orig_wav).unsqueeze(0), model.h
        )

        with torch.inference_mode():
            predicted_wav = model(mel_spectogram).squeeze(0).squeeze(0).numpy()

        predicted_wav = librosa.resample(predicted_wav, orig_sr=sr, target_sr=16000)
        orig_wav = librosa.resample(orig_wav, orig_sr=sr, target_sr=16000)
        pesq_score.append(pesq(16000, orig_wav, predicted_wav, "wb"))

    print(f"Original: {np.mean(pesq_score)} +- {np.std(pesq_score)}")


files = sorted(glob.glob(WAV_DATASET_PATH, recursive=True))[:MAX_FILES]

ours(files)
original(files)
