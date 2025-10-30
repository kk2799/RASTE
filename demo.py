import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Union, Tuple
from beat_this.preprocessing import LogMelSpect
from beat_this.inference import split_piece
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

class AudioFrequencyAnalyzer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.target_sr = 22050
        self.session = self._load_model(model_path, device)
        self.logmel_transform = LogMelSpect(device="cpu")
    
    def _load_model(self, model_path: str, device: str):
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    
    def _aggregate_prediction(self, pred_chunks: list, starts: list, full_size: int, 
                            chunk_size: int, border_size: int, overlap_mode: str) -> Tuple[np.ndarray, np.ndarray]:
        if border_size > 0:
            pred_chunks = [
                {
                    "blade_beat": pchunk["blade_beat"][border_size:-border_size],
                    "shaft_beat": pchunk["shaft_beat"][border_size:-border_size],
                }
                for pchunk in pred_chunks
            ]
        piece_prediction_blade_beat = np.full((full_size,), -1000.0)
        piece_prediction_shaft_beat = np.full((full_size,), -1000.0)
        if overlap_mode == "keep_first":
            pred_chunks = list(reversed(pred_chunks))
            starts = list(reversed(starts))
        for start, pchunk in zip(starts, pred_chunks):
            piece_prediction_blade_beat[
                start + border_size : start + chunk_size - border_size
            ] = pchunk["blade_beat"]
            piece_prediction_shaft_beat[
                start + border_size : start + chunk_size - border_size
            ] = pchunk["shaft_beat"]
        return piece_prediction_blade_beat, piece_prediction_shaft_beat
    
    def _split_predict_aggregate(self, spect: torch.Tensor, chunk_size: int = 3000, 
                               border_size: int = 6, overlap_mode: str = "keep_first") -> dict:
        chunks, starts = split_piece(
            spect, chunk_size, border_size=border_size, avoid_short_end=True
        )
        pred_chunks = []
        input_name = self.session.get_inputs()[0].name
        for chunk in chunks:
            chunk_input = chunk.unsqueeze(0).cpu().numpy().astype(np.float32)
            outputs = self.session.run(None, {input_name: chunk_input})
            pred_chunks.append({
                "blade_beat": outputs[0][0],
                "shaft_beat": outputs[1][0]
            })
        piece_prediction_blade_beat, piece_prediction_shaft_beat = self._aggregate_prediction(
            pred_chunks, starts, spect.shape[0], chunk_size, border_size, overlap_mode
        )
        return {"blade_beat": piece_prediction_blade_beat, "shaft_beat": piece_prediction_shaft_beat}

def demon_spectrum(signal, sample_rate, bandpass_low, bandpass_high, lowpass_cutoff=100):
    from scipy.signal import butter, filtfilt
    nyquist = sample_rate / 2
    b_band, a_band = butter(4, [bandpass_low/nyquist, bandpass_high/nyquist], btype='band')
    bandpass_signal = filtfilt(b_band, a_band, signal)
    envelope = np.abs(bandpass_signal)
    b_low, a_low = butter(4, lowpass_cutoff/nyquist, btype='low')
    demodulated = filtfilt(b_low, a_low, envelope)
    L = len(demodulated)
    fft_result = np.fft.fft(demodulated)
    P = np.abs(fft_result / L)
    P = P[:L // 2]
    P[1:-1] = 2 * P[1:-1]
    freq_axis = np.fft.fftfreq(L, d=1/sample_rate)[:L // 2]
    P[freq_axis < 0.3] = 0
    freq_mask = freq_axis <= 50
    return freq_axis[freq_mask], P[freq_mask]

def compute_fft_spectrum(signal, sample_rate=100):
    L = len(signal)
    fft_result = np.fft.fft(signal)
    P = np.abs(fft_result / L)
    P = P[:L // 2]
    P[1:-1] = 2 * P[1:-1]
    freq_axis = np.fft.fftfreq(L, d=1/sample_rate)[:L // 2]
    P[freq_axis < 0.3] = 0
    freq_mask = freq_axis <= 50
    freq_range = freq_axis[freq_mask]
    P_range = P[freq_mask]
    P_norm = P_range / max(P_range) if max(P_range) > 0 else P_range
    return freq_range, P_norm

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model_path = 'pseudo_demon.onnx'
    audio_path = "test.wav"
    try:
        audio_data, sample_rate = torchaudio.load(audio_path)
        analyzer = AudioFrequencyAnalyzer(model_path, device="cpu")
        waveform = audio_data.flatten() if audio_data.ndim > 1 else audio_data[0]
        waveform = torch.from_numpy(waveform.cpu().numpy().astype(np.float32))
        if sample_rate != analyzer.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, analyzer.target_sr)
            waveform = resampler(waveform)
        mel_spec = analyzer.logmel_transform(waveform)
        model_prediction = analyzer._split_predict_aggregate(mel_spec)
        freq_range, P_blade_beat_norm = compute_fft_spectrum(model_prediction["blade_beat"])
        _, P_shaft_beat_norm = compute_fft_spectrum(model_prediction["shaft_beat"])
        demon_freq, demon_spec = demon_spectrum(
            waveform.numpy(), 
            analyzer.target_sr,
            bandpass_low=3000,
            bandpass_high=5000,
            lowpass_cutoff=100
        )
        demon_spec_norm = demon_spec / max(demon_spec) if max(demon_spec) > 0 else demon_spec
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(freq_range, P_shaft_beat_norm, 'g-', label="shaft_beat")
        axes[0].set_title("shaft_beat Spectrum")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Normalized Magnitude")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].plot(freq_range, P_blade_beat_norm, 'b-', label="blade_beat")
        axes[1].set_title("blade_beat Spectrum")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Normalized Magnitude")
        axes[1].grid(True)
        axes[1].legend()
        axes[2].plot(demon_freq, demon_spec_norm, 'r-', label="DEMON Spectrum")
        axes[2].set_title("DEMON")
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Normalized Magnitude")
        axes[2].grid(True)
        axes[2].legend()
        plt.tight_layout()
        plt.savefig("test.png")
        plt.close()
        print(f"Spectrum saved as: test.png")
    except Exception as e:
        print(f"Audio analysis error: {e}")