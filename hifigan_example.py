import deepaudio
from deepaudio.models.vocoders.hifigan.configuration_hifigan import HifiGANConfig
from deepaudio.models.vocoders.hifigan.modeling_hifigan import HifiGAN, HifiGANPipeline
from deepaudio.pipelines import MelPipeline

hifigan_48k = HifiGAN.from_pretrained("vtb-plan/hifigan-48k")

import torchaudio
import torchaudio.transforms as T

# Load audio
wav, sr = torchaudio.load("zszy_48k.wav")
assert sr == 48000

# mel = mel_spectrogram_torch(wav, 2048, 128, 48000, 512, 2048, 0, None, False)
audio_pipeline = MelPipeline(freq=48000,
                                n_fft=2048,
                                n_mel=128,
                                win_length=2048,
                                hop_length=512)
mel = audio_pipeline(wav)
out = hifigan_48k(mel)

# get output wav