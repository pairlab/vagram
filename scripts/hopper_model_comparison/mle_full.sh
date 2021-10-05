#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 'rtx6000,t4v1,t4v2,p100'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=72:00:00      # time limit
#SBATCH --mem=16GB         # minimum amount of real memory
#SBATCH --job-name=mle_mbrl

source ~/.bashrc
conda activate ClaasICLR

export PYTHONPATH=/h/$USER/mbrl-lib-iclr
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs/mujoco200/bin:/usr/local/nvidia/lib64
export MUJOCO_PY_MUJOCO_PATH=/pkgs/mujoco200
export MUJOCO_PY_MJKEY_PATH=/pkgs/mjpro150/mjkey.txt
export MJLIB_PATH=/pkgs/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=/pkgs/mjpro150/mjkey.txt

cd ~/mbrl-lib-iclr

python3 -m mbrl.examples.main \
	seed=$RANDOM \
	algorithm=mbpo \
	overrides=mbpo_hopper \
	dynamics_model=gaussian_mlp_ensemble \
	hydra.run.dir="$HOME/Claas/$SLURM_JOB_ID"
