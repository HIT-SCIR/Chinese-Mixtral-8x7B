import fire
from transformers import AutoTokenizer


def main(
        tokenizer_name_or_path: str,
        tokenizer_save_path: str,
        additional_tokens: str = "<|beginofutterance|> <|endofutterance|>",
        sequence_length: int = 2048,
        cache_dir: str = "./hf-cache",
):
    additional_tokens = additional_tokens.split(" ")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=cache_dir,
        model_max_length=sequence_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.add_special_tokens(special_tokens_dict={"additional_special_tokens": additional_tokens})

    tokenizer.save_pretrained(tokenizer_save_path)


if __name__ == "__main__":
    fire.Fire(main)
