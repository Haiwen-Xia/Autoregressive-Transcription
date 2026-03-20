from __future__ import annotations

import torch
from audio_understanding.utils import pad_or_truncate
from transformers import AutoTokenizer


class BertOnset:
    r"""BERT tokenizer extended with onset-only transcription tokens.

    Added vocab only contains:
    - time_index=0..6000
    - pitch=0..127
    - drum_pitch=0..127 (optional)

    Event format is strictly:
    [time_index=..., pitch=...] OR [time_index=..., drum_pitch=...]
    where the second token is mutually exclusive.
    """

    def __init__(self, drum_pitch: bool = False) -> None:
        super().__init__()

        self.drum_pitch = drum_pitch
        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")

        new_vocabs = [f"time_index={t}" for t in range(6001)]
        new_vocabs += [f"pitch={p}" for p in range(128)]
        if self.drum_pitch:
            new_vocabs += [f"drum_pitch={p}" for p in range(128)]

        print("Original vocab size: {}".format(len(self.tok)))
        print("New vocab size: {}".format(len(new_vocabs)))
        self.tok.add_tokens(new_vocabs)
        print("Extended vocab size: {}".format(len(self.tok)))

    def texts_to_ids(
        self,
        texts: list[str] | list[list[str]],
        fix_length: int,
    ) -> torch.LongTensor:
        batch_ids = []

        for text in texts:
            if isinstance(text, str):
                tokens = self.tok.tokenize(text)
            elif isinstance(text, list):
                tokens = text
            else:
                raise TypeError(text)

            ids = self.tok.convert_tokens_to_ids(tokens)[0 : fix_length - 2]
            assert ids.count(self.tok.unk_token_id) == 0, (
                "Unknown token is not allowed! Please extend the vocabulary!"
            )

            ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]

            if fix_length:
                ids = pad_or_truncate(ids, fix_length, self.tok.pad_token_id)

            batch_ids.append(ids)

        return torch.LongTensor(batch_ids)

    def __len__(self):
        return len(self.tok)

    @property
    def cls_token_id(self):
        return self.tok.cls_token_id

    @property
    def pad_token_id(self):
        return self.tok.pad_token_id

    @property
    def boa_token_id(self):
        return self.tok.convert_tokens_to_ids("<boa>")

    @property
    def eoa_token_id(self):
        return self.tok.convert_tokens_to_ids("<eoa>")
