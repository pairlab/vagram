echo "Hopper distracted"

r=$RANDOM

echo "MLE 0"
for i in {1..8}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 0
done
echo "MLE 5"
for i in {1..8}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 5
done
echo "MLE 10"
for i in {1..8}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 10
done
echo "MLE 15"
for i in {1..8}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 15
done

echo "VaGraM 0"
for i in {1..8}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 0
done
echo "VaGraM 5"
for i in {1..8}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 5
done
echo "VaGraM 10"
for i in {1..8}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 10
done
echo "VaGraM 15"
for i in {1..8}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 15
done
