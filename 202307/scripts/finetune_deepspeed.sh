#!/bin/bash
#$ -l rt_G.large=1
#$ -l h_rt=4:00:00
#$ -j y
#$ -N finetune-deepspeed
#$ -o logs/
#$ -cwd
#$ -m a
#$ -m b
#$ -m e

source /etc/profile.d/modules.sh
module load cuda/11.8 cudnn/8.9
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export SCRATCH_HOME=/scratch/$(whoami)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# set the wandb project where this run will be logged
export WANDB_PROJECT="rinna"
# save your trained model checkpoint to wandb
export WANDB_LOG_MODEL="true"
# turn off watch to log faster
export WANDB_WATCH="false"

wandb login $WANDB_KEY
deepspeed --num_gpus=$GPUS src/finetune.py --model_name $MODEL --config_file $CONFIG