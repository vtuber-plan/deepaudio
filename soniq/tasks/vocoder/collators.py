# coding=utf-8
"""Collator for vocoder training."""

from typing import Any, Dict, List, Optional
import torch
from soniq.data.collator_base import BaseCollator


class VocoderCollator(BaseCollator):
    """
    Collator for vocoder training.

    Handles padding for variable-length audio and mel spectrograms.
    Also handles segment slicing for training.
    """

    def __init__(
        self,
        config: Any,
        padding_value: float = 0.0,
        batch_first: bool = True,
        segment_size: Optional[int] = None,
    ):
        """
        Initialize VocoderCollator.

        Args:
            config: Configuration object.
            padding_value: Value for padding.
            batch_first: If True, batch is first dimension.
            segment_size: Segment size for training. If None, no slicing.
        """
        super().__init__(
            padding_value=padding_value,
            batch_first=batch_first,
            pad_keys=["audio", "mel"],
        )
        self.config = config
        self.segment_size = segment_size or getattr(config, 'segment_size', 8192)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Randomly samples segments if segment_size is set.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Collated batch with padded tensors.
        """
        # Sample segments if segment_size is set
        if self.segment_size:
            batch = self._sample_segments(batch)

        return super().__call__(batch)

    def _sample_segments(
        self,
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sample random segments from batch items."""
        result = []

        for sample in batch:
            audio = sample.get('audio')
            mel = sample.get('mel')

            if audio is None:
                result.append(sample)
                continue

            audio_len = audio.shape[-1]

            if audio_len > self.segment_size:
                # Random start position
                max_start = audio_len - self.segment_size
                start = torch.randint(0, max_start, (1,)).item()

                # Slice audio
                sample['audio'] = audio[start:start + self.segment_size]

                # Slice mel if available
                if mel is not None:
                    # Compute corresponding mel segment
                    # This depends on hop_size ratio
                    hop_ratio = audio_len / mel.shape[-1]
                    mel_start = int(start / hop_ratio)
                    mel_segment_len = int(self.segment_size / hop_ratio)
                    sample['mel'] = mel[:, mel_start:mel_start + mel_segment_len]

            result.append(sample)

        return result
