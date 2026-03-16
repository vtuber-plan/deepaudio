# coding=utf-8
"""Text symbols for phoneme representation."""

# Phoneme symbols for English
SYMBOLS = [
    "p", "b", "t", "d", "tʃ", "dʒ", "k", "g",  # Stops and affricates
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",  # Fricatives
    "m", "n", "ŋ", "l", "ɹ", "w", "j",  # Nasals and approximants
    "ɛ", "æ", "ɑ", "ɔ", "ʌ", "ə", "i", "u", "e", "o",  # Vowels
    "ɪ", "ʊ", "ɝ", "ɚ",  # Reduced vowels
    "aɪ", "eɪ", "aʊ", "oʊ", "ɔɪ",  # Diphthongs
    "aɪɹ", "aʊɹ", "ɔɪɹ",  # R-colored diphthongs
    ".",  # Syllable break
    "_",  # Word boundary
    " ",  # Space
    "ˈ",  # Primary stress
    "ˌ",  # Secondary stress
]

# Add padding and unknown symbols
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

SPECIAL_SYMBOLS = [PAD, UNK, BOS, EOS]
ALL_SYMBOLS = SPECIAL_SYMBOLS + SYMBOLS

# Create symbol to ID mapping
SYMBOL_TO_ID = {s: i for i, s in enumerate(ALL_SYMBOLS)}
ID_TO_SYMBOL = {i: s for s, i in SYMBOL_TO_ID.items()}

# Phoneme symbol counts
NUM_SYMBOLS = len(ALL_SYMBOLS)
NUM_PHONEMES = len(SYMBOLS)
NUM_SPECIAL = len(SPECIAL_SYMBOLS)


def get_symbol_id(symbol: str) -> int:
    """Get ID for a symbol."""
    return SYMBOL_TO_ID.get(symbol, SYMBOL_TO_ID[UNK])


def get_symbol_by_id(id: int) -> str:
    """Get symbol by ID."""
    return ID_TO_SYMBOL.get(id, UNK)


def text_to_sequence(text: str) -> list:
    """Convert text to sequence of IDs."""
    return [get_symbol_id(s) for s in text]


def sequence_to_text(sequence: list) -> str:
    """Convert sequence of IDs to text."""
    return "".join([get_symbol_by_id(id) for id in sequence])
