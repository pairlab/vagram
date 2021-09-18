#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 't4v1,t4v2'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=72:00:00      # time limit
#SBATCH --mem=16GB         # minimum amount of real memory
#SBATCH --job-name=mle_mbrl
#SBATCH --error=/h/voelcker/logs/vaml_train/%j.err
#SBATCH --output=/h/voelcker/logs/vaml_train/%j.out
#SBATCH --account=deadline
#SBATCH --qos=deadline

source ~/.bashrc
conda activate py37

export PYTHONPATH=/h/$USER/Code/project_codebases/mbrl-lib-shadow-copy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs/mujoco200/bin:/usr/local/nvidia/lib64
export MUJOCO_PY_MUJOCO_PATH=/pkgs/mujoco200
export MUJOCO_PY_MJKEY_PATH=/pkgs/mjpro150/mjkey.txt
export MJLIB_PATH=/pkgs/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=/pkgs/mjpro150/mjkey.txt

cd ~/Code/project_codebases/mbrl-lib-shadow-copy

python3 -m mbrl.examples.main \
	seed=$RANDOM \
	algorithm=mbpo \
	overrides=mbpo_humanoid \
	dynamics_model.model.num_layers=4 \
	dynamics_model.model.hid_size=400 \
	root_dir=/scratch/hdd001/home/voelcker/ \
	hydra.run.dir="/scratch/hdd001/home/voelcker/$SLURM_JOB_ID"
