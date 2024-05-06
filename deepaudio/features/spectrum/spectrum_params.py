
from pydantic import BaseModel

class SpectrumParams(BaseModel):
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int