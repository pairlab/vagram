#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 'rtx6000,t4v1,t4v2,p100'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=120:00:00      # time limit
#SBATCH --mem=16GB         # minimum amount of real memory
#SBATCH --job-name=mle_mbrl
#SBATCH --output=/h/voelcker/logs/vagram-new/%j.out
#SBATCH --error=/h/voelcker/logs/vagram-new/%j.err

hostname

export MUJOCO_PY_BYPASS_LOCK=true

source ~/.bashrc
conda activate ClaasICLR

export PYTHONPATH=/h/$USER/Code/project_codebases/vagram
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/voelcker/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

cd ~/Code/project_codebases/vagram

python3 -m mbrl.examples.main \
	seed=$1 \
	algorithm=mbpo \
	algorithm.double_q_critic.normalize_features=$3 \
	algorithm.diag_gaussian_actor.normalize_features=$3 \
	overrides=mbpo_hopper \
	dynamics_model.model.num_layers=$2 \
	dynamics_model.model.hid_size=64 \
	hydra.run.dir="/checkpoint/voelcker/$SLURM_JOB_ID"
