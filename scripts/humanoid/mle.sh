#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
<<<<<<< HEAD
#SBATCH -p 't4v1,t4v2,rtx6000'           # partition
=======
#SBATCH -p 'p100,t4v1,t4v2,rtx6000'           # partition
>>>>>>> 828c37bf22e0eb7e8ffe9bca426b46c7a69d05c6
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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs/mujoco200/bin:/usr/local/nvidia/lib64
export MUJOCO_PY_MUJOCO_PATH=/pkgs/mujoco200
export MUJOCO_PY_MJKEY_PATH=/pkgs/mjpro150/mjkey.txt
export MJLIB_PATH=/pkgs/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=/pkgs/mjpro150/mjkey.txt

cd ~/Code/project_codebases/mbrl-lib-shadow-copy

python3 -m mbrl.examples.main \
<<<<<<< HEAD
	algorithm=mbpo \
	overrides=mbpo_halfcheetah \
	dynamics_model=gaussian_mlp_ensemble \
	root_dir=/scratch/hdd001/home/voelcker/ \
	hydra.run.dir="/scratch/hdd001/home/voelcker/$SLURM_JOB_ID"
=======
	seed=$RANDOM \
	algorithm=mbpo \
	overrides=mbpo_humanoid \
	dynamics_model.model.num_layers=4 \
	dynamics_model.model.hid_size=400 \
	dynamics_model=gaussian_mlp_ensemble \
	hydra.run.dir="exp/$SLURM_JOB_ID"
>>>>>>> 828c37bf22e0eb7e8ffe9bca426b46c7a69d05c6
