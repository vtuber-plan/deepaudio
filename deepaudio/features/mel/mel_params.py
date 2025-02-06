
from pydantic import BaseModel

from deepaudio.features.spectrum.spectrum_params import SpectrumParams

class MelParams(SpectrumParams):
    mel_channels: int
    mel_fmin: int
    mel_fmax: int