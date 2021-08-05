#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 't4v1,t4v2,rtx6000'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=36:00:00      # time limit 
#SBATCH --mem=32GB         # minimum amount of real memory
#SBATCH --job-name=vaml1
#SBATCH --error=/h/voelcker/logs/vaml_train/%j.err
#SBATCH --output=/h/voelcker/logs/vaml_train/%j.out

source ~/.bashrc
conda activate rl_base

export PYTHONPATH=/h/$USER/Code/project_codebases/pets:/h/$USER/Code/project_codebases/rl-scaffolding
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs/mujoco200/bin:/usr/local/nvidia/lib64
export MUJOCO_PY_MUJOCO_PATH=/pkgs/mujoco200
export MUJOCO_PY_MJKEY_PATH=/pkgs/mjpro150/mjkey.txt
export MJLIB_PATH=/pkgs/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=/pkgs/mjpro150/mjkey.txt

cd ~/Code/project_codebases/NeuralVAML

python -m neural_vaml.main \
	cluster=vector \
	experiment.training.pretrain_iters=0 \
	rl_algorithm=sac \
	model_algorithm=vaml \
	model.hyperparameters.layers="[200,200,200,200]" \
	model.hyperparameters.num_models=8 \
	model.settings.fix_var=True
