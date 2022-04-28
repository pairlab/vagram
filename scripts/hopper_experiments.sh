echo "Hopper distracted"

r=$(( ( RANDOM % 1000000 )  + 1 ))

echo "VaGraM 0"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 0 True
done
echo "VaGraM 0 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 0 False
done
echo "MLE 0"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 0 True
done
echo "MLE 0 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 0 False
done


echo "VaGraM 5"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 5 True
done
echo "VaGraM 5 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 5 False
done
echo "MLE 5"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 5 True
done
echo "MLE 5 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 5 False
done


echo "VaGraM 10"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 10 True
done
echo "VaGraM 10 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 10 False
done
echo "MLE 10"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 10 True
done
echo "MLE 10 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 10 False
done


echo "VaGraM 15"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 15 True
done
echo "VaGraM 15 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/vaml_distr.sh $(expr $r + $i) 15 False
done
echo "MLE 15"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 15 True
done
echo "MLE 15 - no var reduction"
for i in {1..20}; do
	sbatch ./hopper_distraction/mle_distr.sh $(expr $r + $i) 15 False
done