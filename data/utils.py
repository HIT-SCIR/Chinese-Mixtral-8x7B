from typing import List, Dict, Tuple

import tomli

from datasets import load_from_disk, Dataset
import fire


def parse_dataset_name_and_ratio(datasets: List[str]):
    # Parse ["0.1:wudao", "0.2:slimpajama"] into {"wudao": 0.1, "slimpajama": 0.2}
    dataset_name_and_ratio = {}
    for dataset in datasets:
        ratio, name = dataset.split(":")
        dataset_name_and_ratio[name] = float(ratio)
    return dataset_name_and_ratio


def count_token(
        dataset_name2dsNratio: Dict[str, Tuple[Dataset, float]],
        sequence_length: int = 2048,
):
    statistics = {}

    for ds_name, (ds, ratio) in dataset_name2dsNratio.items():
        statistics[ds_name] = len(ds) * ratio * sequence_length / 10 ** 9

    return statistics


def count_all_token(
        sequence_length: int = 2048,
):
    # TODO refactor with `count_token`
    statistics = {}

    with open("./data/datasets.toml", "rb") as f:
        ds_info = tomli.load(f)

    for ds_name, ds in ds_info.items():
        for split in ds["splits"]:
            try:
                ds_path = (ds["root"].format(DATA_DIR="./data", name=ds_name) + "/" +
                           ds["encoded"].format(name=ds_name, split=split))
                print(f"Loading {ds_name} ...")
                dataset = load_from_disk(ds_path)

                statistics[ds_name + "-" + split] = len(dataset) * sequence_length / 10 ** 9
            except FileNotFoundError as e:
                print(e)

    print("=========================================")
    print(statistics)


if __name__ == "__main__":
    fire.Fire(count_all_token)
