#!/usr/bin/bash

#SBATCH -J prep-datasets
#SBATCH -o logs/prep-%j.log
#SBATCH -e logs/prep-%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 56
#SBATCH --mem=512G

. "$HOME"/.miniconda/etc/profile.d/conda.sh
conda activate mistral

python data/preprocess_datasets.py --ds_name SkyPile-150B-2023 --tokenizer_name_or_path tokenizer/Mixtral-8x7B-v0.1-vocab
python data/utils.py
