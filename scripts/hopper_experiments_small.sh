echo "Hopper distracted"

r=$(( ( RANDOM % 1000000 )  + 1 ))

echo "VaGraM 4"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/vaml_layers.sh $(expr $r + $i) 4 True
done
echo "MLE 4"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/mle_layers.sh $(expr $r + $i) 4 False
done
echo "VaGraM 3"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/vaml_layers.sh $(expr $r + $i) 3 True
done
echo "MLE 3"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/mle_layers.sh $(expr $r + $i) 3 False
done
echo "VaGraM 2"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/vaml_layers.sh $(expr $r + $i) 2 True
done
echo "MLE 2"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/mle_layers.sh $(expr $r + $i) 2 False
done

