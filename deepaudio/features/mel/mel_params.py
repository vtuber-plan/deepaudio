
from pydantic import BaseModel

class MelParams(BaseModel):
    mel_channels: int
    mel_fmin: int
    mel_fmax: int