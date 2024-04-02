from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Literal

import tomli
import torch
import torch.distributed
import transformers
from datasets import concatenate_datasets, load_from_disk, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer
from typing_extensions import assert_never

from data.utils import parse_dataset_name_and_ratio, count_token


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model"},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer. "
                          "We distinguish this setting from `model_name_or_path` "
                          "so that datasets will not generate redundant cache."},
    )


@dataclass
class DataArguments:
    train_datasets: List[str] = field(
        default=None,
        metadata={"help": "Training dataset ratio and names. "
                          "Accepted format: 0.1:wudao 0.2:slimpajama, "
                          "which means get 10% data from wudao and 20% data from slimpajama."},
    )
    valid_datasets: List[str] = field(default=None, metadata={"help": "Validation dataset names."})


@dataclass
class PeftArguments:
    enable_lora: bool = field(default=False)
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: Optional[float] = field(default=None)
    lora_rank: Optional[int] = field(default=None)
    lora_target_modules: Optional[List[str]] = field(default=None)
    lora_modules_to_save: Optional[List[str]] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mode: Literal["pt", "sft"] = field(default="pt")
    neftune_noise_alpha: Optional[float] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def build_dataset(train_datasets: List[str], valid_datasets: List[str]):
    with open("./data/datasets.toml", "rb") as f:
        ds_info = tomli.load(f)

    train_data_name2ratio = parse_dataset_name_and_ratio(train_datasets)  # {"wudao": 0.1, "slimpajama": 0.2}
    train_data_name2pathNratio = {}  # {"wudao": ("/path/to/wudao", 0.1), "slimpajama": ("/path/to/slimpajama", 0.2)}
    for ds_name, ratio in train_data_name2ratio.items():
        ds = ds_info[ds_name]
        assert "train" in ds["splits"]
        train_data_name2pathNratio[ds_name] = (ds["root"].format(DATA_DIR="./data", name=ds_name) + "/" +
                                               ds["encoded"].format(name=ds_name, split="train"),
                                               ratio)

    valid_data_name2path = {}  # {"wudao": "/path/to/wudao", "slimpajama": "/path/to/slimpajama"}
    for ds_name in valid_datasets:
        ds = ds_info[ds_name]
        assert "dev" in ds["splits"]
        valid_data_name2path[ds_name] = (ds["root"].format(DATA_DIR="./data", name=ds_name) + "/" +
                                         ds["encoded"].format(name=ds_name, split="dev"))

    train_data_name2dsNratio = {ds_name: (load_from_disk(path), ratio)  # {"wudao": (wudao_dataset, 0.1), ...}
                                for ds_name, (path, ratio) in train_data_name2pathNratio.items()}
    train_dataset = concatenate_datasets(
        [ds.select(range(int(len(ds) * ratio))) for ds, ratio in train_data_name2dsNratio.values()]
    )  # 0.1 * wudao_dataset + 0.2 * slimpajama_dataset
    valid_dataset = {name: load_from_disk(path) for name, path in valid_data_name2path.items()}

    print_rank_0("=========================================")
    print_rank_0(f"Training dataset: {count_token(train_data_name2dsNratio)}")
    print_rank_0(f"Validation dataset: {valid_dataset.keys()}")
    print_rank_0("=========================================")

    return train_dataset, valid_dataset


def build_sft_dataset(train_datasets: List[str], valid_datasets: List[str]):
    # TODO refactor with build_dataset

    with open("./data/datasets.toml", "rb") as f:
        ds_info = tomli.load(f)

    train_data_name2ratio = parse_dataset_name_and_ratio(train_datasets)  # {"wudao": 0.1, "slimpajama": 0.2}
    train_data_name2pathNratio = {}  # {"wudao": ("/path/to/wudao", 0.1), "slimpajama": ("/path/to/slimpajama", 0.2)}
    for ds_name, ratio in train_data_name2ratio.items():
        ds = ds_info[ds_name]
        assert "train" in ds["splits"]
        train_data_name2pathNratio[ds_name] = (ds["root"].format(DATA_DIR="./data", name=ds_name) + "/" +
                                               ds["doc"].format(name=ds_name, split="train") + ".jsonl",
                                               ratio)

    valid_data_name2path = {}  # {"wudao": "/path/to/wudao", "slimpajama": "/path/to/slimpajama"}
    for ds_name in valid_datasets:
        ds = ds_info[ds_name]
        assert "dev" in ds["splits"]
        valid_data_name2path[ds_name] = (ds["root"].format(DATA_DIR="./data", name=ds_name) + "/" +
                                         ds["doc"].format(name=ds_name, split="dev") + ".jsonl")

    load_jsonl_dataset = partial(load_dataset, path="json", split="train", cache_dir="./hf-cache")
    train_data_name2dsNratio = {
        ds_name: (load_jsonl_dataset(data_files=path), ratio)  # {"wudao": (wudao_dataset, 0.1), ...}
        for ds_name, (path, ratio) in train_data_name2pathNratio.items()
    }
    train_dataset = concatenate_datasets(
        [ds.select(range(int(len(ds) * ratio))) for ds, ratio in train_data_name2dsNratio.values()]
    )  # 0.1 * wudao_dataset + 0.2 * slimpajama_dataset
    valid_dataset = {name: load_jsonl_dataset(data_files=path) for name, path in valid_data_name2path.items()}

    print_rank_0("=========================================")
    print_rank_0(f"Training dataset: {len(train_dataset)}")
    print_rank_0(f"Validation dataset: {valid_dataset}")
    print_rank_0("=========================================")

    return train_dataset, valid_dataset


def print_rank_0(*args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def train():
    model_args: ModelArguments
    data_args: DataArguments
    peft_args: PeftArguments
    training_args: TrainingArguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype="auto",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="</s>"),
            tokenizer=tokenizer,
            model=model,
        )

    if peft_args.enable_lora:
        assert peft_args.lora_alpha
        assert peft_args.lora_dropout
        assert peft_args.lora_rank
        assert peft_args.lora_target_modules
        assert peft_args.lora_modules_to_save

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=peft_args.lora_target_modules,
            modules_to_save=peft_args.lora_modules_to_save,
            inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
        )

        # Prevent no-grad problem.
        # See https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        # and https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641/2
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, peft_config)

        # To resume from existing adapter
        # from peft import PeftModel
        # model = PeftModel.from_pretrained(model, "/path/to/adapter/checkpoint", is_trainable=True)

        for name, param in model.named_parameters():
            if "lora" in name:
                param.data = param.data.to(torch.bfloat16)

        print_rank_0("=========================================")
        model.print_trainable_parameters()
        print_rank_0("=========================================")

    if training_args.mode == "pt":
        builder = build_dataset
    elif training_args.mode == "sft":
        builder = build_sft_dataset
    else:
        assert_never(None)
    train_dataset, valid_dataset = builder(data_args.train_datasets, data_args.valid_datasets)

    model.is_parallelizable = True
    model.model_parallel = True

    if training_args.mode == "pt":
        if training_args.neftune_noise_alpha:
            print_rank_0("WARNING: `neftune_noise_alpha` is not supported in `pt` mode.")
        partial_trainer = partial(Trainer)
    elif training_args.mode == "sft":
        partial_trainer = partial(
            SFTTrainer,
            max_seq_length=training_args.model_max_length,
            neftune_noise_alpha=training_args.neftune_noise_alpha,
            peft_config=peft_config if peft_args.enable_lora else None,
            dataset_text_field="text",
        )
    else:
        assert_never(None)

    trainer = partial_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=training_args,
    )
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
