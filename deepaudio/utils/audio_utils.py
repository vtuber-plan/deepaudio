


import numpy as np
import torch
import torchaudio


def save_audio(path: str, waveform: np.ndarray, fs: int, add_silence=False, turn_up=False, volume_peak=0.9):
    """Save audio to path with processing  (turn up volume, add silence)
    Args:
        path (str): path to save audio
        waveform (numpy array): waveform to save
        fs (int): sampling rate
        add_silence (bool, optional): whether to add silence to beginning and end. Defaults to False.
        turn_up (bool, optional): whether to turn up volume. Defaults to False.
        volume_peak (float, optional): volume peak. Defaults to 0.9.
    """
    if turn_up:
        # continue to turn up to volume_peak
        ratio = volume_peak / max(waveform.max(), abs(waveform.min()))
        waveform = waveform * ratio

    if add_silence:
        silence_len = fs // 20
        silence = np.zeros((silence_len,), dtype=waveform.dtype)
        result = np.concatenate([silence, waveform, silence])
        waveform = result

    waveform = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if len(waveform.size()) == 1:
        waveform = waveform[None, :]
    elif waveform.size(0) != 1:
        # Stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    torchaudio.save(path, waveform, fs, encoding="PCM_S", bits_per_sample=16)
