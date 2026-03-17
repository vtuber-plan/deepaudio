# coding=utf-8
"""
Phone extractor for Soniq.
"""

from typing import List, Optional, Union, Dict
from soniq.text.g2p import GraphemeToPhoneme
from soniq.text.symbols import SYMBOL_TO_ID, ID_TO_SYMBOL, PAD, UNK, BOS, EOS


class PhoneExtractor:
    """
    Phone (phoneme) extractor for text-to-speech.

    This class converts text to phoneme sequences and provides
    encoding/decoding functionality.

    Example:
        ```python
        extractor = PhoneExtractor()
        phones = extractor.extract("hello world")
        ids = extractor.encode(phones)
        ```
    """

    def __init__(
        self,
        language: str = "en",
        add_bos: bool = False,
        add_eos: bool = False,
        use_symbol_to_id: bool = True,
    ):
        """
        Initialize PhoneExtractor.

        Args:
            language: Language code.
            add_bos: Whether to add beginning-of-sequence token.
            add_eos: Whether to add end-of-sequence token.
            use_symbol_to_id: Whether to use the global SYMBOL_TO_ID mapping.
        """
        self.language = language
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.use_symbol_to_id = use_symbol_to_id
        self.g2p = GraphemeToPhoneme(language=language)

    def extract(
        self,
        text: str,
        word_level: bool = False,
    ) -> Union[str, List[str]]:
        """
        Extract phonemes from text.

        Args:
            text: Input text.
            word_level: If True, return list of phonemes per word.

        Returns:
            Phoneme string or list of phoneme strings.
        """
        return self.g2p(text, word_level=word_level)

    def encode(
        self,
        phonemes: Union[str, List[str]],
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode phonemes to IDs.

        Args:
            phonemes: Phoneme string or list of phonemes.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of phoneme IDs.
        """
        # Convert phoneme string to list
        if isinstance(phonemes, str):
            # Split by space and filter empty strings
            phoneme_list = [p for p in phonemes.split() if p]
        else:
            phoneme_list = phonemes

        # Convert to IDs
        if self.use_symbol_to_id:
            ids = [SYMBOL_TO_ID.get(p, SYMBOL_TO_ID[UNK]) for p in phoneme_list]
        else:
            # Custom mapping can be added here
            ids = list(range(len(phoneme_list)))

        # Add special tokens
        if add_special_tokens:
            if self.add_bos:
                ids = [SYMBOL_TO_ID[BOS]] + ids
            if self.add_eos:
                ids = ids + [SYMBOL_TO_ID[EOS]]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode phoneme IDs to string.

        Args:
            ids: List of phoneme IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Phoneme string.
        """
        special_ids = {
            SYMBOL_TO_ID[PAD],
            SYMBOL_TO_ID[UNK],
            SYMBOL_TO_ID[BOS],
            SYMBOL_TO_ID[EOS],
        }

        phonemes = []
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            phonemes.append(ID_TO_SYMBOL.get(id, UNK))

        return " ".join(phonemes)

    def __call__(
        self,
        text: str,
        return_ids: bool = True,
    ) -> Union[str, Dict[str, Union[str, List[int]]]]:
        """
        Process text to phonemes.

        Args:
            text: Input text.
            return_ids: Whether to also return phoneme IDs.

        Returns:
            Phoneme string or dict with phonemes and IDs.
        """
        phonemes = self.extract(text)

        if return_ids:
            ids = self.encode(phonemes)
            return {
                "phonemes": phonemes,
                "ids": ids,
            }
        else:
            return phonemes
