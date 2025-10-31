import numpy as np
import torch
import torchaudio

class LogMelSpect(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=441,
        hop_length=221,
        f_min=500,
        f_max=10000,
        n_mels=96,
        mel_scale="slaney",
        normalized="frame_length",
        power=1,
        log_multiplier=1000,
        device="cpu",
        normalize_audio=True,
    ):
        super().__init__()
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        ).to(device)
        self.log_multiplier = log_multiplier
        self.normalize_audio = normalize_audio

    def forward(self, x):
        if self.normalize_audio and x.numel() > 0:
            mean = x.mean()
            std = x.std()
            x = (x - mean) / (std + 1e-6)               
        return torch.log1p(self.log_multiplier * self.spect_class(x).T)
