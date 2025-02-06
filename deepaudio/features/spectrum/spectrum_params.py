
from pydantic import BaseModel

class SpectrumParams(BaseModel):
    n_fft: int
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int