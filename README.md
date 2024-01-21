# deepaudio
Advanced machine learning speech models based on PyTorch.

Given the success of Transformers in the field of NLP and its provision of convenient APIs and tools for easily downloading and training state-of-the-art pre-trained models, deepaudio serves as a complementary addition in the domain of speech. It provides a similar interface and can be published on the Hugging Face Hub. The implemented model types in deepaudio include:

* ASR (Automatic Speech Recognition)
* TTS (Text-to-Speech): VITS
* Vocoder: HifiGAN, MelGAN
* F0
* Content Encoder
* Speaker Encoder


## Installation

### Method 1: With pip

```bash
pip install deepaudio
```

or:

```bash
pip install git+https://github.com/vtuber-plan/deepaudio.git 
```

### Method 2: From source

1. Clone this repository
```bash
git clone https://github.com/vtuber-plan/deepaudio.git
cd deepaudio
```

2. Install the Package
```bash
pip install --upgrade pip
pip install .
```

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

## License
deepaudio is under the MIT License.
