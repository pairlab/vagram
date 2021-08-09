echo "Hyperparameter Test on Cheetah"

echo "Normal"
sbatch hp_comparison/full.sh
sbatch hp_comparison/full.sh
sbatch hp_comparison/full.sh
sbatch hp_comparison/full.sh

echo "No clipping"
sbatch hp_comparison/no_clip.sh
sbatch hp_comparison/no_clip.sh
sbatch hp_comparison/no_clip.sh
sbatch hp_comparison/no_clip.sh

echo "No target"
sbatch hp_comparison/target_vf.sh
sbatch hp_comparison/target_vf.sh
sbatch hp_comparison/target_vf.sh
sbatch hp_comparison/target_vf.sh

echo "No target, no clipping"
sbatch hp_comparison/target_vf_no_clip.sh
sbatch hp_comparison/target_vf_no_clip.sh
sbatch hp_comparison/target_vf_no_clip.sh
sbatch hp_comparison/target_vf_no_clip.sh
