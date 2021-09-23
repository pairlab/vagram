#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=36:00:00      # time limit
#SBATCH --mem=8GB         # minimum amount of real memory
#SBATCH --job-name=mle_mbrl
#SBATCH --error=~/logs/vaml_train/%j.err
#SBATCH --output=~/logs/vaml_train/%j.out

source ~/.bashrc
conda activate py37

export PYTHONPATH=$HOME/Code/project_codebases/mbrl-lib-shadow-copy

cd ~/Code/project_codebases/mbrl-lib-shadow-copy

python3 -m mbrl.examples.main \
	seed=$RANDOM \
	algorithm=mbpo \
	overrides=mbpo_hopper \
	dynamics_model=gaussian_mlp_ensemble \
	hydra.run.dir="exp/$SLURM_JOB_ID"
