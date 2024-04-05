__all__ = [
    "SentencePieceTokenizer",
    "MistralTokenizer",
    "LlamaTokenizer",
]

import sentencepiece as sp

from typing import List, Optional

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
            val = self.spp.IdToPiece(key)
            # NB: hard coded.
            if key == 3 + 0x0a:
                val = "\n"
            elif ord(val[0]) == 9601:
                val2 = [" "]
                for k in range(1, len(val)):
                    if ord(val[k]) != 9601:
                        val2.append(val[k:])
                        break
                    val2.append(" ")
                val = "".join(val2)
            return val
        if isinstance(key, str):
            idx = self.spp.PieceToId(key)
            # NB: pad token index.
            if idx == 0 and key != self._unk_piece():
                raise IndexError
            return idx
        raise IndexError

    def __call__(self, text: str, prepend: Optional[bool] = True) -> List[int]:
        if prepend is None or not prepend:
            #tok = self.spp.Encode("<p>{}".format(text), add_bos = True)
            #assert tok[:4] == [1, 523, 28720, 28767]
            tok = self.spp.Encode("0{}".format(text), add_bos = True)
            if tok[:3] != [1, 28705, 28734]:
                print("DEBUG:  SentencePieceTokenizer: tok = {}".format(tok[:5]))
            assert tok[:3] == [1, 28705, 28734]
            tok = tok[3:]
        else:
            tok = self.spp.Encode(text, add_bos = True)
        return tok

MistralTokenizer = SentencePieceTokenizer
LlamaTokenizer = SentencePieceTokenizer
