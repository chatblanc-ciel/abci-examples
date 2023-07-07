#!/bin/bash
#$ -l rt_G.large=1
#$ -j y
#$ -N finetune
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/11.8 cudnn/8.9
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python src/finetune.py --model_name $MODEL --config_file $CONFIG