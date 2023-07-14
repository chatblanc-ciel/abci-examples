#!/bin/bash
#$ -l rt_F=30
#$ -l h_rt=4:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N finetune-deepspeed-multinode
#$ -o logs/
#$ -cwd
#$ -m a
#$ -m b
#$ -m e

source /etc/profile.d/modules.sh
module load cuda/11.8 cudnn/8.9 nccl/2.16 hpcx/2.12
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export SCRATCH_HOME=/scratch/$(whoami)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MASTER_ADDR=$HOSNAME

# set the wandb project where this run will be logged
export WANDB_PROJECT="rinna"
# save your trained model checkpoint to wandb
export WANDB_LOG_MODEL="true"
# turn off watch to log faster
export WANDB_WATCH="false"

# login in wandb for logging
wandb login $WANDB_KEY

# create hostfile
hostfile=$(mktemp)
for l in `cat $SGE_JOB_HOSTLIST`; do echo $l slots=4; done > $hostfile
trap "rm $hostfile" EXIT
trap "trap - EXIT; rm $hostifle; exit -1" INT PIPE TERM

deepspeed \
  --master_addr $HOSTNAME \
  --hostfile $hostfile \
  --no_ssh_check \
  --launcher OpenMPI \
  src/finetune.py \
  --model_name $MODEL \
  --config_file $CONFIG