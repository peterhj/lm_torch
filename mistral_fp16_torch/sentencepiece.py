__all__ = [
    "SentencePieceTokenizer",
    "MistralTokenizer",
    "LlamaTokenizer",
]

import sentencepiece as sp

from typing import List

class SentencePieceTokenizer:
    def __init__(self, path = None):
        self.spp = sp.SentencePieceProcessor(model_file = path)
        self.unk_piece = None
        self.vocab_len = None

    def _unk_piece(self) -> str:
        if self.unk_piece is None:
            # FIXME: pad token index.
            self.unk_piece = self.spp.IdToPiece(0)
        return self.unk_piece

    def _vocab_len(self) -> int:
        if self.vocab_len is None:
            self.vocab_len = self.spp.piece_size()
        return self.vocab_len

    def __len__(self) -> int:
        return self._vocab_len()

    def __getitem__(self, key: int | str) -> str | int:
        if isinstance(key, int):
            return self.spp.IdToPiece(key)
        if isinstance(key, str):
            idx = self.spp.PieceToId(key)
            # FIXME: pad token index.
            if idx == 0 and key != self._unk_piece():
                raise IndexError
            return idx
        raise IndexError

    def __call__(self, text: str) -> List[int]:
        return self.spp.Encode(text, add_bos = True)

MistralTokenizer = SentencePieceTokenizer
LlamaTokenizer = SentencePieceTokenizer
