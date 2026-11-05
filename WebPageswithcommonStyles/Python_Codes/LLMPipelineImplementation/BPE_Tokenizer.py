"""
bpe_tokenizer_numpy.py
========================
A from-scratch Byte-Pair Encoding (BPE) tokenizer — the same algorithm
GPT-2's tokenizer uses — built with ONLY NumPy and the Python standard
library. No `tokenizers`, no `transformers`.

This replaces:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

with a tokenizer you train yourself on your own corpus, whose merge
rules and vocabulary you can inspect line by line.

BPE algorithm (Sennrich et al., 2016 — the same core idea used by
GPT-2/GPT-3/RoBERTa's tokenizers):
    1. Start with every word split into individual characters.
    2. Count all adjacent character-pair frequencies across the corpus.
    3. Merge the single most frequent pair into one new symbol.
    4. Repeat steps 2-3 until you've learned `vocab_size` merges.
    5. To encode new text, apply the learned merges in the same order
       they were learned.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

END_OF_WORD = "</w>"  # marks word boundaries, same convention as GPT-2's tokenizer


class BPETokenizer:
    """A Byte-Pair Encoding tokenizer trained from scratch on a corpus."""

    def __init__(self, vocab_size: int = 1000) -> None:
        """
        Args:
            vocab_size: Target vocabulary size (base characters + learned
                merges). Training stops early if the corpus runs out of
                pairs to merge before reaching this size.
        """
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []          # ordered list of learned merges
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, corpus: List[str]) -> None:
        """Learn BPE merge rules from a list of raw text documents.

        Args:
            corpus: List of raw text strings to train on.
        """
        # Step 1: pre-tokenize into words, split each word into characters
        # with an end-of-word marker, and count word frequencies.
        word_freqs: Counter[str] = Counter()
        for doc in corpus:
            for word in re.findall(r"\S+", doc.lower()):
                word_freqs[word] += 1

        # Represent each word as a tuple of symbols (starts as characters).
        splits: Dict[str, List[str]] = {
            word: list(word) + [END_OF_WORD] for word in word_freqs
        }

        # Base vocabulary = every individual character seen, plus END_OF_WORD.
        base_vocab = sorted({sym for symbols in splits.values() for sym in symbols})

        num_merges_needed = max(0, self.vocab_size - len(base_vocab))

        for _ in range(num_merges_needed):
            pair_freqs = self._count_pair_frequencies(splits, word_freqs)
            if not pair_freqs:
                break  # no more pairs left to merge

            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self._merge_pair(best_pair, splits)
            self.merges.append(best_pair)

        # Final vocabulary = base characters + every merged symbol produced.
        final_symbols = sorted(
            {sym for symbols in splits.values() for sym in symbols} | set(base_vocab)
        )
        self.token_to_id = {tok: i for i, tok in enumerate(final_symbols)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    @staticmethod
    def _count_pair_frequencies(
        splits: Dict[str, List[str]], word_freqs: Counter
    ) -> Counter[Tuple[str, str]]:
        """Count frequency of every adjacent symbol pair across the corpus."""
        pair_freqs: Counter[Tuple[str, str]] = Counter()
        for word, symbols in splits.items():
            freq = word_freqs[word]
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i + 1])] += freq
        return pair_freqs

    @staticmethod
    def _merge_pair(
        pair: Tuple[str, str], splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Merge every occurrence of `pair` into a single symbol."""
        merged_symbol = "".join(pair)
        new_splits: Dict[str, List[str]] = {}

        for word, symbols in splits.items():
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_splits[word] = new_symbols

        return new_splits

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def _apply_merges_to_word(self, word: str) -> List[str]:
        """Apply learned merges, in learned order, to a single word."""
        symbols = list(word) + [END_OF_WORD]

        for pair in self.merges:
            merged_symbol = "".join(pair)
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str) -> np.ndarray:
        """Encode raw text into an array of token ids.

        Args:
            text: Raw input string.

        Returns:
            A 1-D NumPy array of integer token ids (matches the dtype
            used elsewhere in llm_pretraining.py's tensors before they're
            converted to torch tensors).
        """
        ids: List[int] = []
        for word in re.findall(r"\S+", text.lower()):
            for symbol in self._apply_merges_to_word(word):
                # Unknown symbols (unseen at training time) fall back to
                # a reserved <unk> id if present, else are skipped.
                ids.append(self.token_to_id.get(symbol, self.token_to_id.get("<unk>", 0)))
        return np.array(ids, dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        """Decode an array of token ids back into a text string.

        Args:
            ids: 1-D array/list of integer token ids.

        Returns:
            The reconstructed string (word boundaries restored from
            END_OF_WORD markers).
        """
        symbols = [self.id_to_token[int(i)] for i in ids]
        text = "".join(symbols).replace(END_OF_WORD, " ")
        return text.strip()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        """Save vocabulary and merge rules to a JSON file."""
        payload = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load a previously trained tokenizer from a JSON file."""
        payload = json.loads(Path(path).read_text())
        tok = cls(vocab_size=payload["vocab_size"])
        tok.merges = [tuple(pair) for pair in payload["merges"]]
        tok.token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        return tok


if __name__ == "__main__":
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog sleeps while the quick fox runs",
        "pretraining a language model requires a large text corpus",
    ]

    tokenizer = BPETokenizer(vocab_size=80)
    tokenizer.train(corpus)

    print(f"Learned {len(tokenizer.merges)} merges")
    print("First 10 merges:", tokenizer.merges[:10])

    sample = "the quick fox"
    ids = tokenizer.encode(sample)
    print(f"\nEncoded '{sample}' -> {ids}")
    print(f"Decoded back -> '{tokenizer.decode(ids)}'")