"""
Text tokenizers for SMILES and protein sequences.

Four tokenization strategies, from simplest to most sophisticated:

1. char_level    — one character = one token, fixed vocabulary
2. atom_level    — chemistry-aware: multi-char atoms (Br, Cl, [NH2+], etc.) as single tokens
3. bpe           — Byte-Pair Encoding trained on the dataset (using HuggingFace tokenizers)
4. wordpiece     — WordPiece trained on the dataset (using HuggingFace tokenizers)

Each tokenizer exposes:
  .encode(text)         → List[int]   (token ids)
  .vocab_size           → int
  .pad_id               → int
  .unk_id               → int

A shared collate_fn pads batches for DataLoader use.
"""

import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

TOKENIZER_CACHE = Path("cache/tokenizers")
TOKENIZER_CACHE.mkdir(parents=True, exist_ok=True)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Character-level tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Maps every unique character to an integer ID.
    Vocabulary is built from a corpus or supplied explicitly.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_corpus(cls, texts: List[str]) -> "CharTokenizer":
        chars = sorted(set("".join(texts)))
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
        for ch in chars:
            if ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids = [self.vocab.get(ch, self.vocab[UNK_TOKEN]) for ch in text]
        if add_special:
            ids = [self.vocab[BOS_TOKEN]] + ids + [self.vocab[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.inv_vocab.get(i, UNK_TOKEN) for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        with open(path) as f:
            return cls(json.load(f))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Atom-level SMILES tokenizer
# ─────────────────────────────────────────────────────────────────────────────

# Regex pattern: matches multi-char atoms first, then single chars
_SMILES_ATOM_RE = re.compile(
    r"(\[[^\]]+\]"          # bracketed atoms e.g. [NH2+], [C@@H]
    r"|Br|Cl|Si|Se|As"      # two-char atoms
    r"|[BCNOPSFIbcnops]"    # single-char organic subset
    r"|[0-9]"               # ring closure digits
    r"|[=#@%\+\-\\\/()\.])" # bond/structure chars
)


class AtomLevelTokenizer(CharTokenizer):
    """
    Chemistry-aware SMILES tokenizer.
    Splits SMILES into atom-level tokens (Br, Cl, [NH2+] are single tokens).
    Falls back to character-level for unknown patterns.
    """

    @classmethod
    def from_corpus(cls, smiles_list: List[str]) -> "AtomLevelTokenizer":
        all_tokens: set[str] = set()
        for smi in smiles_list:
            all_tokens.update(cls._tokenize_smiles(smi))
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
        for tok in sorted(all_tokens):
            if tok not in vocab:
                vocab[tok] = len(vocab)
        inst = cls(vocab)
        return inst

    @staticmethod
    def _tokenize_smiles(smi: str) -> List[str]:
        return _SMILES_ATOM_RE.findall(smi)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        tokens = self._tokenize_smiles(text)
        ids = [self.vocab.get(t, self.vocab[UNK_TOKEN]) for t in tokens]
        if add_special:
            ids = [self.vocab[BOS_TOKEN]] + ids + [self.vocab[EOS_TOKEN]]
        return ids


# ─────────────────────────────────────────────────────────────────────────────
# 3. BPE tokenizer (trained on dataset)
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained on a corpus using HuggingFace tokenizers.
    Works for both SMILES and protein sequences.
    """

    def __init__(self, hf_tokenizer=None):
        self._tok = hf_tokenizer

    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 1000,
              name: str = "bpe") -> "BPETokenizer":
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import CharDelimiterSplit

        cache_path = TOKENIZER_CACHE / f"{name}_vs{vocab_size}.json"
        if cache_path.exists():
            print(f"  [tokenizer] Loading cached BPE from {cache_path}")
            tok = Tokenizer.from_file(str(cache_path))
            return cls(tok)

        # Truncate training texts to 512 chars to avoid OOM on long protein sequences.
        # BPE only needs to learn sub-word patterns — full sequences not required.
        train_texts = [t[:512] for t in texts]
        print(f"  [tokenizer] Training BPE (vocab_size={vocab_size}) on {len(train_texts):,} texts …")
        tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tok.pre_tokenizer = CharDelimiterSplit(" ")  # treat whole string as one word

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
            min_frequency=2,
        )
        tok.train_from_iterator(train_texts, trainer=trainer)
        tok.save(str(cache_path))
        print(f"  [tokenizer] BPE vocab size: {tok.get_vocab_size()}")
        return cls(tok)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        enc = self._tok.encode(text)
        ids = enc.ids
        if add_special:
            bos = self._tok.token_to_id(BOS_TOKEN)
            eos = self._tok.token_to_id(EOS_TOKEN)
            ids = [bos] + ids + [eos]
        return ids

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id(UNK_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# 4. WordPiece tokenizer (trained on dataset)
# ─────────────────────────────────────────────────────────────────────────────

class WordPieceTokenizer:
    """
    WordPiece tokenizer trained on a corpus using HuggingFace tokenizers.
    """

    def __init__(self, hf_tokenizer=None):
        self._tok = hf_tokenizer

    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 1000,
              name: str = "wordpiece") -> "WordPieceTokenizer":
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import CharDelimiterSplit

        cache_path = TOKENIZER_CACHE / f"{name}_vs{vocab_size}.json"
        if cache_path.exists():
            print(f"  [tokenizer] Loading cached WordPiece from {cache_path}")
            tok = Tokenizer.from_file(str(cache_path))
            return cls(tok)

        train_texts = [t[:512] for t in texts]
        print(f"  [tokenizer] Training WordPiece (vocab_size={vocab_size}) on {len(train_texts):,} texts …")
        tok = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
        tok.pre_tokenizer = CharDelimiterSplit(" ")

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
            min_frequency=2,
        )
        tok.train_from_iterator(train_texts, trainer=trainer)
        tok.save(str(cache_path))
        print(f"  [tokenizer] WordPiece vocab size: {tok.get_vocab_size()}")
        return cls(tok)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        enc = self._tok.encode(text)
        ids = enc.ids
        if add_special:
            bos = self._tok.token_to_id(BOS_TOKEN)
            eos = self._tok.token_to_id(EOS_TOKEN)
            ids = [bos] + ids + [eos]
        return ids

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id(UNK_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build tokenizer by name
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer(name: str, train_texts: List[str]):
    """
    name examples:
      "smiles_char"           → CharTokenizer on SMILES
      "smiles_atom"           → AtomLevelTokenizer on SMILES
      "smiles_bpe_512"        → BPE vocab=512 on SMILES
      "smiles_bpe_1000"       → BPE vocab=1000 on SMILES
      "protein_char"          → CharTokenizer on protein seqs
      "protein_bpe_512"       → BPE vocab=512 on protein seqs
      "protein_wordpiece_512" → WordPiece vocab=512 on protein seqs
    """
    cache_pkl = TOKENIZER_CACHE / f"{name}.pkl"
    if cache_pkl.exists():
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)

    parts = name.split("_")
    kind  = "_".join(parts[1:])   # e.g. "char", "atom", "bpe_512"

    if kind == "char":
        tok = CharTokenizer.from_corpus(train_texts)
    elif kind == "atom":
        tok = AtomLevelTokenizer.from_corpus(train_texts)
    elif kind.startswith("bpe"):
        vocab_size = int(kind.split("_")[1]) if "_" in kind else 1000
        tok = BPETokenizer.train(train_texts, vocab_size=vocab_size, name=name)
    elif kind.startswith("wordpiece"):
        vocab_size = int(kind.split("_")[1]) if "_" in kind else 1000
        tok = WordPieceTokenizer.train(train_texts, vocab_size=vocab_size, name=name)
    else:
        raise ValueError(f"Unknown tokenizer kind: {kind}")

    with open(cache_pkl, "wb") as f:
        pickle.dump(tok, f)
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Collation helpers for DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def pad_sequence(sequences: List[List[int]], pad_id: int,
                 max_len: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad a list of token ID lists to equal length.
    Returns (padded_array, attention_mask) both shape (N, L).
    """
    L = max_len or max(len(s) for s in sequences)
    N = len(sequences)
    out  = np.full((N, L), pad_id, dtype=np.int64)
    mask = np.zeros((N, L), dtype=np.float32)
    for i, seq in enumerate(sequences):
        trunc = seq[:L]
        out[i, :len(trunc)] = trunc
        mask[i, :len(trunc)] = 1.0
    return out, mask
