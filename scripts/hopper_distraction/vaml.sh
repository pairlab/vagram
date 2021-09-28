#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 't4v1,t4v2,p100,rtx6000'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=72:00:00      # time limit
#SBATCH --mem=16GB         # minimum amount of real memory
#SBATCH --job-name=mle_mbrl
#SBATCH --error=/h/voelcker/logs/vaml_train/%j.err
#SBATCH --output=/h/voelcker/logs/vaml_train/%j.out

source ~/.bashrc
conda activate py37

export PYTHONPATH=/h/$USER/Code/project_codebases/mbrl-lib-shadow-copy-2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs/mujoco200/bin:/usr/local/nvidia/lib64
export MUJOCO_PY_MUJOCO_PATH=/pkgs/mujoco200
export MUJOCO_PY_MJKEY_PATH=/pkgs/mjpro150/mjkey.txt
export MJLIB_PATH=/pkgs/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=/pkgs/mjpro150/mjkey.txt

cd ~/Code/project_codebases/mbrl-lib-shadow-copy-2

python3 -m mbrl.examples.main \
	seed=$RANDOM \
	algorithm=mbpo \
	overrides=mbpo_hopper \
	hydra.run.dir="exp/$SLURM_JOB_ID"
