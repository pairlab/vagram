echo "Hopper distracted"

r=$RANDOM

echo "MLE 4"
for i in {1..8}; do
	sbatch ./hopper_model_comparison/mle_full.sh $(expr $r + $i)
done
echo "MLE 3"
for i in {1..8}; do
	sbatch ./hopper_model_comparison/mle_3_64.sh $(expr $r + $i)
done
echo "MLE 2"
for i in {1..8}; do
	sbatch ./hopper_model_comparison/mle_2_64.sh $(expr $r + $i)
done

echo "VaGraM 4"
for i in {1..8}; do
	sbatch ./hopper_model_comparison/vaml_full.sh $(expr $r + $i)
done
echo "VaGraM 3"
for i in {1..8}; do
	sbatch ./hopper_model_comparison/vaml_3_64.sh $(expr $r + $i)
done
echo "VaGraM 2"
for i in {1..8}; do
	sbatch ./hopper_model_comparison/vaml_2_64.sh $(expr $r + $i)
done
