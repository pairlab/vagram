echo "Hopper MBPO VAML"

echo "VAML large"
sbatch hopper/vaml.sh
sbatch hopper/vaml.sh
sbatch hopper/vaml.sh
sbatch hopper/vaml.sh
sbatch hopper/vaml.sh
sbatch hopper/vaml.sh
sbatch hopper/vaml.sh
sbatch hopper/vaml.sh

echo "VAML small"
sbatch hopper/vaml_small.sh
sbatch hopper/vaml_small.sh
sbatch hopper/vaml_small.sh
sbatch hopper/vaml_small.sh
sbatch hopper/vaml_small.sh
sbatch hopper/vaml_small.sh
sbatch hopper/vaml_small.sh
sbatch hopper/vaml_small.sh

echo "MLE large"
sbatch hopper/mle.sh
sbatch hopper/mle.sh
sbatch hopper/mle.sh
sbatch hopper/mle.sh
sbatch hopper/mle.sh
sbatch hopper/mle.sh
sbatch hopper/mle.sh
sbatch hopper/mle.sh

echo "MLE small"
sbatch hopper/mle_small.sh
sbatch hopper/mle_small.sh
sbatch hopper/mle_small.sh
sbatch hopper/mle_small.sh
sbatch hopper/mle_small.sh
sbatch hopper/mle_small.sh
sbatch hopper/mle_small.sh
sbatch hopper/mle_small.sh
