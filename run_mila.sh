#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=1-00:00
#SBATCH -c 8
#SBATCH --mem 48G 
#SBATCH --gres=gpu:rtx8000:1
module load singularity/3.7.1
# on mila
# put your sif image here (better on a salloc'ed node):
# export SINGULARITY_CACHEDIR=$SLURM_TMPDIR/.singularity
# module load singularity 
# cd $SCRATCH
# singularity pull docker://artemzholus/jax_cuda11.8
./job.sh "$@"