import pkg_resources
from transformers import LlamaTokenizer


class LlamaBPEDropoutTokenizer(LlamaTokenizer):
    def __init__(self, *args, **kwargs):
        if "bpe_dropout_alpha" in kwargs:
            # We override `_tokenize` method from transformers==4.40.0
            pkg_resources.require(["transformers==4.40.0"])

            self.bpe_dropout_alpha = kwargs.pop("bpe_dropout_alpha")

        super().__init__(*args, **kwargs)

    def _tokenize(self, text, **kwargs):
        """
        Override LlamaTokenizer._tokenize to add BPE Dropout to the tokenization process.
        Require transformers==4.40.0
        """
        SPIECE_UNDERLINE = "▁"

        tokens = self.sp_model.encode(
            text,
            alpha=self.bpe_dropout_alpha if self.bpe_dropout_alpha is not None else 0,
            enable_sampling=self.bpe_dropout_alpha is not None,
            out_type=str,
        )
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens
