echo "DM control comparison"

r=$(( ( RANDOM % 1000000 )  + 1 ))

# echo "VaGraM cheetah"
# for i in {1..8}; do
# 	sbatch ./dm_control_comparison/vaml_env.sh $(expr $r + $i) mbpo_halfcheetah False
# done
# echo "MLE cheetah"
# for i in {1..16}; do
# 	sbatch ./dm_control_comparison/mle_env.sh $(expr $r + $i) mbpo_halfcheetah False
# done
echo "VaGraM ant"
for i in {1..8}; do
	sbatch ./dm_control_comparison/vaml_env.sh $(expr $r + $i) mbpo_ant False
done
# echo "MLE ant"
# for i in {1..16}; do
# 	sbatch ./dm_control_comparison/mle_env.sh $(expr $r + $i) mbpo_ant False
# done
# echo "VaGraM walker"
# for i in {1..8}; do
# 	sbatch ./dm_control_comparison/vaml_env.sh $(expr $r + $i) mbpo_walker False
# done
# echo "MLE walker"
# for i in {1..16}; do
# 	sbatch ./dm_control_comparison/mle_env.sh $(expr $r + $i) mbpo_walker False
# done
# echo "VaGraM humanoid"
# for i in {1..8}; do
# 	sbatch ./dm_control_comparison/vaml_env.sh $(expr $r + $i) mbpo_humanoid False
# done
# echo "MLE humanoid"
# for i in {1..8}; do
# 	sbatch ./dm_control_comparison/mle_env.sh $(expr $r + $i) mbpo_humanoid False
# done

