# deepaudio
State-of-the-art Audio Machine Learning Models.

## Usage

Use HifiGAN to convert mel to wav:
```python
import torchaudio
import torchaudio.transforms as T

# Load audio
wav, sr = torchaudio.load("zszy_48k.wav")
assert sr == 48000

from deepaudio.pipelines import MelPipeline
audio_pipeline = MelPipeline(freq=48000, n_fft=2048, n_mel=128, win_length=2048, hop_length=512)

from deepaudio.models.vocoders.hifigan.configuration_hifigan import HifiGANConfig
from deepaudio.models.vocoders.hifigan.modeling_hifigan import HifiGAN, HifiGANPipeline

hifigan_48k = HifiGAN.from_pretrained("vtb-plan/hifigan-48k")

mel = audio_pipeline(wav)
out = hifigan_48k(mel)

```