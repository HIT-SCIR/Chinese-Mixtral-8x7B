#!/usr/bin/bash

#SBATCH -J mixtral
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
#SBATCH -p gpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --mem=800G
#SBATCH --gres=gpu:8

. "$HOME"/.miniconda/etc/profile.d/conda.sh
conda activate vocab-ext

TRAIN_DATASETS=(
    1:your-dataset-1
    1:your-dataset-2
)

VALID_DATASETS=(
    your-val-dataset-1
    your-val-dataset-2
)

TRAIN_PARAMS=""
TRAIN_PARAMS+=" --mode sft"
TRAIN_PARAMS+=" --enable_lora"
TRAIN_PARAMS+=" --lora_alpha 128"
TRAIN_PARAMS+=" --lora_dropout 0.05"
TRAIN_PARAMS+=" --lora_rank 64"
TRAIN_PARAMS+=" --lora_target_modules q_proj v_proj k_proj o_proj w1 w2 w3"
TRAIN_PARAMS+=" --lora_modules_to_save embed_tokens lm_head"
TRAIN_PARAMS+=" --ddp_find_unused_parameters=False"
TRAIN_PARAMS+=" --model_name_or_path /path/to/model"
TRAIN_PARAMS+=" --tokenizer_name_or_path /path/to/tokenizer"
TRAIN_PARAMS+=" --neftune_noise_alpha 5"
TRAIN_PARAMS+=" --bpe_dropout_alpha 0.1"
TRAIN_PARAMS+=" --train_datasets ${TRAIN_DATASETS[*]}"
TRAIN_PARAMS+=" --valid_datasets ${VALID_DATASETS[*]}"
TRAIN_PARAMS+=" --dataloader_drop_last"
TRAIN_PARAMS+=" --cache_dir hf-cache"
TRAIN_PARAMS+=" --output_dir outputs/$SLURM_JOB_ID"
TRAIN_PARAMS+=" --num_train_epochs 1"
TRAIN_PARAMS+=" --model_max_length 32768"
TRAIN_PARAMS+=" --per_device_train_batch_size 1"
TRAIN_PARAMS+=" --gradient_accumulation_steps 20"
TRAIN_PARAMS+=" --adam_beta1 0.9"
TRAIN_PARAMS+=" --adam_beta2 0.999"
TRAIN_PARAMS+=" --adam_epsilon 1e-8"
TRAIN_PARAMS+=" --weight_decay 0"
TRAIN_PARAMS+=" --per_device_eval_batch_size 1"
TRAIN_PARAMS+=" --evaluation_strategy steps"
TRAIN_PARAMS+=" --eval_steps 200"
TRAIN_PARAMS+=" --save_strategy steps"
TRAIN_PARAMS+=" --save_steps 500"
TRAIN_PARAMS+=" --learning_rate 1e-5"
TRAIN_PARAMS+=" --warmup_ratio 0.05"
TRAIN_PARAMS+=" --logging_strategy steps"
TRAIN_PARAMS+=" --logging_steps 1"
TRAIN_PARAMS+=" --lr_scheduler_type cosine"
TRAIN_PARAMS+=" --report_to wandb"
TRAIN_PARAMS+=" --run_name $SLURM_JOB_ID"
TRAIN_PARAMS+=" --gradient_checkpointing"
TRAIN_PARAMS+=" --bf16"

TORCHRUN_PARAMS='--nproc_per_node 8 --node_rank=$SLURM_NODEID --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=0 --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT'
srun --label --export=ALL bash -c "
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1);
    MASTER_PORT=55511;
    torchrun $TORCHRUN_PARAMS train.py $TRAIN_PARAMS
"
