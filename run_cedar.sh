#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=3-00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem 100000M
#SBATCH --gpus-per-node=v100l:1
module load singularity/3.8
# on cedar
# put your sif image here (better on a salloc'ed node):
# export SINGULARITY_CACHEDIR=$SLURM_TMPDIR/.singularity
# module load singularity 
# cd $SCRATCH
# singularity pull docker://artemzholus/jax_cuda11.8
export WANDB_MODE=offline
export WANDB_ENTITY=chandar-rl
export WANDB_PROJECT=s4dreamer
./job.sh "$@"
