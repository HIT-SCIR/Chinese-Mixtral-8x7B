import os
import fire
import json
from pathlib import Path
from tqdm.auto import tqdm
import pyarrow.parquet as pq


def parse_file(filename):
    pds = pq.read_pandas(filename, columns=None).to_pandas()
    return pds.to_json(
        path_or_buf=None,
        orient="records",
        lines=True,
        date_format="iso",
        date_unit="us",
        compression="gzip",
    ).split("\n")


def main(
    target_jsonl: str = "./data/en/DKYoon-SlimPajama-6B.jsonl",
    root: str = "./data/en/DKYoon-SlimPajama-6B/data",
):
    root = Path(root)
    target_jsonl = Path(target_jsonl)

    files = os.listdir(root)

    with open(target_jsonl, "w") as f:
        for file in tqdm(files):
            objs = parse_file(root / file)
            for obj_str in list(objs):
                try:
                    obj = json.loads(obj_str)
                    f.write(f"{json.dumps(obj, ensure_ascii=False)}\n")
                except:
                    print("json parse error")


if __name__ == "__main__":
    fire.Fire(main)
