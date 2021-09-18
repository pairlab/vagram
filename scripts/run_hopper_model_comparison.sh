echo "Hopper MBPO VAML"

echo "2 64"
sbatch hopper_model_comparison/mle_2_64.sh
sbatch hopper_model_comparison/mle_2_64.sh
sbatch hopper_model_comparison/mle_2_64.sh
sbatch hopper_model_comparison/mle_2_64.sh

echo "2 128"
sbatch hopper_model_comparison/mle_2_128.sh
sbatch hopper_model_comparison/mle_2_128.sh
sbatch hopper_model_comparison/mle_2_128.sh
sbatch hopper_model_comparison/mle_2_128.sh

echo "3 64"
sbatch hopper_model_comparison/mle_3_64.sh
sbatch hopper_model_comparison/mle_3_64.sh
sbatch hopper_model_comparison/mle_3_64.sh
sbatch hopper_model_comparison/mle_3_64.sh

echo "4 64"
sbatch hopper_model_comparison/mle_4_64.sh
sbatch hopper_model_comparison/mle_4_64.sh
sbatch hopper_model_comparison/mle_4_64.sh
sbatch hopper_model_comparison/mle_4_64.sh
