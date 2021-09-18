echo "HalfCheetah MBPO VAML"

echo "VAML large"
sbatch cheetah/vaml.sh
sbatch cheetah/vaml.sh
sbatch cheetah/vaml.sh
sbatch cheetah/vaml.sh
sbatch cheetah/vaml.sh
sbatch cheetah/vaml.sh
sbatch cheetah/vaml.sh
sbatch cheetah/vaml.sh

echo "VAML small"
sbatch cheetah/vaml_small.sh
sbatch cheetah/vaml_small.sh
sbatch cheetah/vaml_small.sh
sbatch cheetah/vaml_small.sh
sbatch cheetah/vaml_small.sh
sbatch cheetah/vaml_small.sh
sbatch cheetah/vaml_small.sh
sbatch cheetah/vaml_small.sh

echo "MLE large"
sbatch cheetah/mle.sh
sbatch cheetah/mle.sh
sbatch cheetah/mle.sh
sbatch cheetah/mle.sh
sbatch cheetah/mle.sh
sbatch cheetah/mle.sh
sbatch cheetah/mle.sh
sbatch cheetah/mle.sh

echo "MLE small"
sbatch cheetah/mle_small.sh
sbatch cheetah/mle_small.sh
sbatch cheetah/mle_small.sh
sbatch cheetah/mle_small.sh
sbatch cheetah/mle_small.sh
sbatch cheetah/mle_small.sh
sbatch cheetah/mle_small.sh
sbatch cheetah/mle_small.sh
