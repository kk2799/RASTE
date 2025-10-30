import numpy as np
import torch
import torchaudio

class LogMelSpect(torch.nn.Module):
    """
    对音频波形计算对数梅尔频谱图的PyTorch模块
    
    参数:
        sample_rate: 采样率 (默认 22050 Hz)
        n_fft: FFT窗口大小 (默认 1024)
        hop_length: 帧移 (默认 441)
        f_min: 最低频率 (默认 30 Hz)
        f_max: 最高频率 (默认 11000 Hz)
        n_mels: 梅尔滤波器组数量 (默认 128)
        mel_scale: 梅尔刻度类型 (默认 "slaney")
        normalized: 归一化方式 (默认 "frame_length")
        power: 频谱图的幂次 (默认 1)
        log_multiplier: 对数变换前的乘数 (默认 1000)
        device: 计算设备 (默认 "cpu")
        normalize_audio: 是否对输入音频进行标准化 (默认 True)
    """
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
        """
        计算输入音频的对数梅尔频谱图
        
        参数:
            x: 形状为(T,)的一维音频波形数组
            
        返回:
            形状为(F,128)的二维对数梅尔频谱图，
            其中F是时间帧数，128是梅尔频率箱的数量
        """
        # 对音频波形进行标准化处理 (减均值除以标准差)
        if self.normalize_audio and x.numel() > 0:
            mean = x.mean()
            std = x.std()
            # 避免除以零
            x = (x - mean) / (std + 1e-6)
                
        return torch.log1p(self.log_multiplier * self.spect_class(x).T)
