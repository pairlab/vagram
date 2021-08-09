#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 't4v1,t4v2,rtx6000'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=36:00:00      # time limit
#SBATCH --mem=32GB         # minimum amount of real memory
#SBATCH --job-name=mle_mbrl
#SBATCH --error=/h/voelcker/logs/vaml_train/%j.err
#SBATCH --output=/h/voelcker/logs/vaml_train/%j.out

source ~/.bashrc
conda activate py37

export PYTHONPATH=/h/$USER/Code/project_codebases/mbrl-lib-shadow-copy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64

cd ~/Code/project_codebases/mbrl-lib-shadow-copy

python3 -m mbrl.examples.main \
	seed=$RANDOM \
	algorithm=mbpo \
	overrides=mbpo_walker \
	dynamics_model=gaussian_mlp_ensemble \
	root_dir="/scratch/gobi2/voelcker/exp"
