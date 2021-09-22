echo "Hopper MBPO VAML"

echo "2 64 VAML"
sbatch hopper_model_comparison/vaml_2_64.sh
sbatch hopper_model_comparison/vaml_2_64.sh
sbatch hopper_model_comparison/vaml_2_64.sh
sbatch hopper_model_comparison/vaml_2_64.sh

echo "2 128 VAML"
sbatch hopper_model_comparison/vaml_2_128.sh
sbatch hopper_model_comparison/vaml_2_128.sh
sbatch hopper_model_comparison/vaml_2_128.sh
sbatch hopper_model_comparison/vaml_2_128.sh

echo "3 64 VAML"
sbatch hopper_model_comparison/vaml_3_64.sh
sbatch hopper_model_comparison/vaml_3_64.sh
sbatch hopper_model_comparison/vaml_3_64.sh
sbatch hopper_model_comparison/vaml_3_64.sh

echo "2 64 MLE"
sbatch hopper_model_comparison/mle_2_64.sh
sbatch hopper_model_comparison/mle_2_64.sh
sbatch hopper_model_comparison/mle_2_64.sh
sbatch hopper_model_comparison/mle_2_64.sh

echo "2 128 MLE"
sbatch hopper_model_comparison/mle_2_128.sh
sbatch hopper_model_comparison/mle_2_128.sh
sbatch hopper_model_comparison/mle_2_128.sh
sbatch hopper_model_comparison/mle_2_128.sh

echo "3 64 MLE"
sbatch hopper_model_comparison/mle_3_64.sh
sbatch hopper_model_comparison/mle_3_64.sh
sbatch hopper_model_comparison/mle_3_64.sh
sbatch hopper_model_comparison/mle_3_64.sh
