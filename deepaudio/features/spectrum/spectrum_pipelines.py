
import torch
import torchaudio
import torchaudio.transforms as T

from deepaudio.features.spectrum.spectrum_params import SpectrumParams

class SpectrumPipeline(torch.nn.Module):
    def __init__(self, params: SpectrumParams):
        super().__init__()
        self.params = params

        pad = int((params.n_fft-params.hop_length)/2)
        self.spec = T.Spectrogram(
            n_fft=params.n_fft, 
            win_length=params.win_length, 
            hop_length=params.hop_length,
            pad=pad,
            power=None,
            center=False,
            pad_mode='reflect',
            normalized=False,
            onesided=True
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        return spec
