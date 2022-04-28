echo "DM control comparison"

r=$(( ( RANDOM % 1000000 )  + 1 ))

echo "VaGraM cheetah"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/vaml_full.sh $(expr $r + $i) mbpo_halfcheetah True
done
echo "MLE cheetah"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/mle_full.sh $(expr $r + $i) mbpo_halfcheetah False
done
echo "VaGraM ant"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/vaml_3_64.sh $(expr $r + $i) mbpo_ant True
done
echo "MLE ant"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/mle_3_64.sh $(expr $r + $i) mbpo_ant False
done
echo "VaGraM walker"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/vaml_2_64.sh $(expr $r + $i) mbpo_walker True
done
echo "MLE walker"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/mle_2_64.sh $(expr $r + $i) mbpo_walker False
done
echo "VaGraM humanoid"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/vaml_2_64.sh $(expr $r + $i) mbpo_humanoid True
done
echo "MLE humanoid"
for i in {1..20}; do
	sbatch ./hopper_model_comparison/mle_2_64.sh $(expr $r + $i) mbpo_humanoid False
done

