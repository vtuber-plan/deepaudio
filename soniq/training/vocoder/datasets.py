# coding=utf-8
"""Vocoder dataset and collator."""

from typing import Any, Callable, Dict, List, Optional
import torch
import os

from soniq.data import ManifestDataset, BaseCollator
from soniq.processing.audio import load_audio
from soniq.processing.features.mel import MelSpectrogramExtractor


class VocoderDataset(ManifestDataset):
    """
    Dataset for vocoder training.

    Loads audio files and extracts mel spectrograms on-the-fly.

    Example manifest entry:
    {
        "id": "utt_001",
        "audio_path": "/path/to/audio.wav",
        "duration": 3.5,
        "speaker": "spk_001"
    }
    """

    def __init__(
        self,
        manifest_path: str,
        config: Any,
        sample_rate: Optional[int] = None,
        use_spkid: bool = False,
    ):
        """
        Initialize VocoderDataset.

        Args:
            manifest_path: Path to manifest JSON/JSONL file.
            config: Configuration object.
            sample_rate: Target sample rate.
            use_spkid: Whether to use speaker IDs.
        """
        self.config = config
        self.sample_rate = sample_rate or getattr(config, 'sample_rate', 24000)
        self.use_spkid = use_spkid

        # Create readers
        # Note: reader key "audio_path" matches the manifest key
        readers = {
            "audio_path": lambda path: self._load_audio(path),
        }

        # Create processors (mel extraction)
        self.mel_extractor = MelSpectrogramExtractor(config)

        # Filter function (remove very short or long samples)
        max_duration = getattr(config, 'max_audio_duration', 10.0)
        min_duration = getattr(config, 'min_audio_duration', 0.1)

        filter_fn = lambda x: min_duration <= x.get('duration', 0) <= max_duration

        super().__init__(
            manifest_path=manifest_path,
            readers=readers,
            processors=[self._process_sample],
            filter_fn=filter_fn,
        )

        # Load speaker mapping if needed
        if use_spkid:
            self.spk2id = self._load_spk2id()
        else:
            self.spk2id = None

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        return load_audio(path, self.sample_rate, mono=True)

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a sample: extract mel spectrogram and sample segment."""
        # Get audio from audio_path reader result
        audio = sample.get('audio_path')
        if audio is not None:
            # Rename audio_path to audio for consistency
            sample['audio'] = audio
            del sample['audio_path']

            # Extract mel spectrogram
            sample = self.mel_extractor(sample)

            # Sample a random segment for training
            audio_length = audio.shape[-1]
            segment_size = self.segment_size

            if audio_length > segment_size:
                # Random start for training
                max_start = audio_length - segment_size
                start = torch.randint(0, max_start, (1,)).item()
                sample['audio'] = audio[:, start:start + segment_size]
                # Re-extract mel from the segmented audio
                sample = self.mel_extractor(sample)
            elif audio_length < segment_size:
                # Pad if audio is too short
                sample['audio'] = torch.nn.functional.pad(audio, (0, segment_size - audio_length))
                sample = self.mel_extractor(sample)

            # Add audio length
            sample['audio_length'] = sample['audio'].shape[-1]
            if 'mel' in sample:
                sample['mel_length'] = sample['mel'].shape[-1]

        # Add speaker ID if needed
        if self.use_spkid and self.spk2id is not None:
            speaker = sample.get('speaker')
            if speaker:
                sample['spk_id'] = self.spk2id.get(speaker, 0)

        return sample

    @property
    def segment_size(self) -> int:
        """Get segment size from config."""
        return getattr(self.config, 'segment_size', 8192)

    def _load_spk2id(self) -> Dict[str, int]:
        """Load speaker-to-ID mapping."""
        # Try to load from manifest directory
        manifest_dir = os.path.dirname(self.manifest_path)
        spk2id_path = os.path.join(manifest_dir, 'spk2id.json')

        if os.path.exists(spk2id_path):
            import json
            with open(spk2id_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item by index."""
        sample = super().__getitem__(index)

        # Ensure audio is shape (1, time)
        if 'audio' in sample:
            audio = sample['audio']
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            sample['audio'] = audio

        return sample


class VocoderCollator(BaseCollator):
    """
    Collator for vocoder training.

    Handles padding for variable-length audio and mel spectrograms.
    """

    def __init__(
        self,
        config: Any,
        padding_value: float = 0.0,
        batch_first: bool = True,
    ):
        """
        Initialize VocoderCollator.

        Args:
            config: Configuration object.
            padding_value: Value for padding.
            batch_first: If True, batch is first dimension.
        """
        super().__init__(
            padding_value=padding_value,
            batch_first=batch_first,
            pad_keys=["mel"],  # Only pad mel, handle audio separately
        )
        self.config = config
        self.segment_size = getattr(config, 'segment_size', 8192)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Collated batch with padded tensors.
        """
        # Handle audio separately - stack instead of pad
        # Audio should be (batch, 1, time) shape
        audio_list = [s['audio'] for s in batch if 'audio' in s]
        mel_list = [s['mel'] for s in batch if 'mel' in s]

        # Stack audio (all should have same length after segment sampling)
        if audio_list:
            # Ensure all audio have same length by padding to max
            max_len = max(a.shape[-1] for a in audio_list)
            audio_padded = []
            for audio in audio_list:
                if audio.shape[-1] < max_len:
                    # Pad on the right
                    pad_len = max_len - audio.shape[-1]
                    audio_padded.append(torch.nn.functional.pad(audio, (0, pad_len)))
                else:
                    audio_padded.append(audio)
            batch_audio = torch.stack(audio_padded)  # (batch, 1, time)
        else:
            batch_audio = torch.zeros(0, 1, 0)

        # Pad mel spectrograms to have same length
        # mel shape is (n_mel, time), we need to pad the time dimension
        if mel_list:
            max_mel_len = max(m.shape[-1] for m in mel_list)
            mel_padded = []
            for mel in mel_list:
                if mel.shape[-1] < max_mel_len:
                    pad_len = max_mel_len - mel.shape[-1]
                    mel_padded.append(torch.nn.functional.pad(mel, (0, pad_len)))
                else:
                    mel_padded.append(mel)
            batch_mel = torch.stack(mel_padded)  # (batch, n_mel, time)
        else:
            batch_mel = torch.zeros(0, 0, 0)

        # Collect other fields
        batch_dict = {
            'audio': batch_audio,
            'mel': batch_mel,
        }

        # Add lengths
        batch_dict['audio_lengths'] = torch.tensor([a.shape[-1] for a in audio_list])
        batch_dict['mel_lengths'] = torch.tensor([m.shape[-1] for m in mel_list])

        # Add IDs and other metadata
        if 'id' in batch[0]:
            batch_dict['ids'] = [s['id'] for s in batch]

        return batch_dict
