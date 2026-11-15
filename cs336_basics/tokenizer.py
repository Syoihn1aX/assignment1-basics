from __future__ import annotations

import regex
import os
from collections.abc import Iterable, Iterator
from typing import Any

from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.token_to_id = {b: i for i, b in self.vocab.items()}

        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.merged = {pair: pair[0] + pair[1] for pair in self.merge_ranks}

        self.special_tokens = [t for t in (special_tokens or []) if t]
        self.special_to_id = {}
        next_id = max(self.vocab)+1 if self.vocab else 0
        for token in sorted(self.special_tokens, key=len, reverse=True):
            data = token.encode("utf-8")
            if data in self.token_to_id:
                tid = self.token_to_id[data]
            else:
                while next_id in self.vocab:
                    next_id += 1

                tid = next_id
                self.vocab[tid] = data
                self.token_to_id[data] = tid
                next_id += 1
            self.special_to_id[token] = tid

        if self.special_to_id:
            alts = "|".join(
                regex.escape(t) for t in sorted(self.special_to_id, key=len, reverse=True) 
            )
            self._special_re = regex.compile(f"({alts})")
            self._max_special_len = max(len(t) for t in self.special_to_id)
        else:
            self._special_re =None
            self._max_special_len = 0

        self._cache = {}

    def encode(self, text):
        if not text:
            return []
        if self._special_re is None:
            return self._encode_ordinary(text)

        ids = []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            sid = self.special_to_id.get(piece)
            if sid is not None:
                ids.append(sid)
            else:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def _encode_ordinary(self, text):
        ids = []
        for m in regex.finditer(PAT, text):
            ids.extend(self._encode_pretoken(m.group()))
        return ids

    def _encode_pretoken(self, piece):
        tokens = self._bpe(piece.encode("utf-8"))
        return [self.token_to_id[t] for t in tokens]

    def _bpe(self, data):
        cached = self._cache.get(data)
        if cached is not None:
            return cached

        symbols = [bytes([b]) for b in data]
        while len(symbols) > 1:
            best_rank = None
            best_pair = None
            for left, right in zip(symbols, symbols[1:]):
                rank = self.merge_ranks.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (left, right)
            if best_pair is None:
                break
            symbols = self._merge_all(symbols, best_pair)

        if len(self._cache) > 8192:
            self._cache.clear()
        self._cache[data] = symbols
        return symbols

    def _merge_all(self, symbols, pair):
        left, right = pair
        merged = self.merged[pair]
        out = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            cut = self._safe_cut(buffer)
            if cut:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    def _safe_cut(self, buffer):
        n = len(buffer)
        if self._special_re is None:
            cut = n
        else:
            cut = n -self._max_special_len
            if cut <= 0:
                return 0
            for m in self._special_re.finditer(buffer):
                if m.start() < cut < m.end():
                    cut = m.end()

        base = 0
        if self._special_re is not None:
            for m in self._special_re.finditer(buffer, 0, cut):
                base = m.end()
        seg = buffer[base:cut]
        if not seg:
            return cut
        last_start = 0
        for m in regex.finditer(PAT, seg):
            last_start = m.start()
        cut = base + last_start

        while cut > 0 and regex.fullmatch(r"\s", buffer[cut - 1]):
            cut -= 1
        return cut

def split_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]

    pattern = regex.compile(
        "(" + "|".join(
            regex.escape(token)
            for token in sorted(
                special_tokens, key=len, reverse=True
            )
        ) + ")"
    )

    return [part for part in pattern.split(text) if part]

def train_bpe(input_path, vocab_size, special_tokens):
    raise NotImplementedError

def get_tokenizer(vocab, merges, special_tokens=None) -> Any:
    return Tokenizer(vocab, merges, special_tokens)