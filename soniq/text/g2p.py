# coding=utf-8
"""
Grapheme-to-Phoneme conversion for Soniq.
"""

from typing import List, Optional, Union
import re


class GraphemeToPhoneme:
    """
    Grapheme-to-Phoneme (G2P) converter.

    This class provides basic G2P conversion functionality.
    For production use, consider using a more robust library like:
    - g2p_en (English)
    - phonemizer (Multi-language)
    - pypinyin (Chinese)

    Example:
        ```python
        g2p = GraphemeToPhoneme()
        phonemes = g2p("hello world")
        print(phonemes)  # "h ə l oʊ w ɝ l d"
        ```
    """

    def __init__(self, language: str = "en"):
        """
        Initialize G2P converter.

        Args:
            language: Language code ("en" for English).
        """
        self.language = language
        self.dictionary = self._load_dictionary()

    def _load_dictionary(self) -> dict:
        """Load pronunciation dictionary."""
        # Simple rule-based G2P for demonstration
        # In production, this should use a proper dictionary like CMUDict
        return {
            "a": "eɪ",
            "b": "b",
            "c": "k",
            "d": "d",
            "e": "i",
            "f": "f",
            "g": "g",
            "h": "h",
            "i": "aɪ",
            "j": "dʒ",
            "k": "k",
            "l": "l",
            "m": "m",
            "n": "n",
            "o": "oʊ",
            "p": "p",
            "q": "kw",
            "r": "ɹ",
            "s": "s",
            "t": "t",
            "u": "ju",
            "v": "v",
            "w": "w",
            "x": "ks",
            "y": "j",
            "z": "z",
        }

    def __call__(self, text: str, word_level: bool = False) -> Union[str, List[str]]:
        """
        Convert text to phonemes.

        Args:
            text: Input text.
            word_level: If True, return list of phonemes per word.

        Returns:
            Phoneme string or list of phoneme strings.
        """
        # Clean text
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)

        # Split into words
        words = text.split()

        # Convert each word to phonemes
        phonemes = []
        for word in words:
            word_phonemes = self._word_to_phonemes(word)
            phonemes.append(word_phonemes)

        if word_level:
            return phonemes
        else:
            return " ".join(phonemes)

    def _word_to_phonemes(self, word: str) -> str:
        """
        Convert a single word to phonemes.

        Args:
            word: Input word.

        Returns:
            Phoneme string for the word.
        """
        # Simple letter-to-phoneme conversion
        # This is a naive implementation - use CMUDict in production
        phonemes = []
        for char in word:
            if char in self.dictionary:
                phonemes.append(self.dictionary[char])
            else:
                phonemes.append(char)

        return " ".join(phonemes)

    def phonemize(self, text: str) -> str:
        """Alias for __call__."""
        return self(text)
