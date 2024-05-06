


import torch

from deepaudio.features.spectrum.spectrum_params import SpectrumParams

class SpectrumPipeline(torch.nn.Module):
    def __init__(self, params: SpectrumParams):
        super().__init__()

        self.freq=freq

        pad = int((n_fft-hop_length)/2)
        self.spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)

        # self.strech = T.TimeStretch(hop_length=hop_length, n_freq=freq)
        self.spec_aug = torch.nn.Sequential(
            GaussianNoise(min_snr=0.0001, max_snr=0.02),
            T.FrequencyMasking(freq_mask_param=80),
            # T.TimeMasking(time_mask_param=80),
        )

        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor, aug: bool=False) -> torch.Tensor:
        shift_waveform = waveform
        # Convert to power spectrogram
        spec = self.spec(shift_waveform)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        # Apply SpecAugment
        if aug:
            spec = self.spec_aug(spec)
        # Convert to mel-scale
        mel = self.mel_scale(spec)
        return mel

def extract_linear_features(y, cfg, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    hann_window[str(y.device)] = torch.hann_window(cfg.win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.n_fft - cfg.hop_size) / 2), int((cfg.n_fft - cfg.hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_size,
        win_length=cfg.win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.squeeze(spec, 0)
    return spec

