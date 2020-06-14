
########################################################################################################################
# These experiments are to generate predictions on test data

# ++++++++++++++++++++++++++++++++++ Baselines ++++++++++++++++++++++++++++++++++
# Test time

bsub -o base-ts.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--test --method ct_both --save_dir experiments/multi-baselines/bimodal --exp_name ts

bsub -o base-ts.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--test --method ct_both_attn --save_dir experiments/multi-baselines/bimodal_attn --exp_name ts

bsub -o base-ts.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--test --method ct --save_dir experiments/multi-baselines/ct --exp_name ts

bsub -o base-ts.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--test --method ct_attn --save_dir experiments/multi-baselines/ct_attn --exp_name ts

# ++++++++++++++++++++++++++++++++++ Bimodal ++++++++++++++++++++++++++++++++++
declare -a fractions=(0 1 2 3 4 5)

for j in "${fractions[@]}";
do bsub -o fra-ts.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--test --method ct_both --save_dir experiments/pag/fractions/frac_"$j" --exp_name ts
done

########################################################################################################################

# These experiments are to generate predictions (on valid and test) for all the cross-validation folds
# ++++++++++++++++++++++++++++++++++ Baselines (16) ++++++++++++++++++++++++++++++++++

declare -a folds=(0 1 2 3)
for i in "${folds[@]}";
do bsub -o base-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method ct --save_dir /home/karthikp/multi-baselines/ct --exp_name cv"$i"
done

for i in "${folds[@]}";
do bsub -o base-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method ct_attn --save_dir /home/karthikp/multi-baselines/ct_attn --exp_name cv"$i"
done

for i in "${folds[@]}";
do bsub -o base-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method ct_both --save_dir /home/karthikp/multi-baselines/bimodal --exp_name cv"$i"
done

for i in "${folds[@]}";
do bsub -o base-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method ct_both_attn --save_dir /home/karthikp/multi-baselines/bimodal_attn --exp_name cv"$i"
done

declare -a fractions=(0 1 2 3 4 5)
# ++++++++++++++++++++++++++++++++++ Bimodal (28) ++++++++++++++++++++++++++++++++++

for j in "${fractions[@]}";
do
  for i in "${folds[@]}";
  do bsub -o fra-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
  --valid --test --method ct_both --save_dir /home/karthikp/pag/fractions/frac_"$j" --exp_name cv"$i"
  done
done

for i in "${folds[@]}";
do bsub -o fra-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method ct_both --save_dir /home/karthikp/pag/fractions/complete --exp_name cv"$i"
done


# ++++++++++++++++++++++++++++++++++ PAG (30*2) ++++++++++++++++++++++++++++++++++
for j in "${fractions[@]}";
do
  for i in "${folds[@]}";
  do bsub -o fra-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
  --valid --test --method pet_attn --save_dir /home/karthikp/pag/less/frac_"$j" --exp_name cv"$i"
  done
done

for i in "${folds[@]}";
do bsub -o fra-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method pet_attn --save_dir /home/karthikp/pag/less/complete --exp_name cv"$i"
done

bsub -o fra-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method pet_attn --save_dir /home/karthikp/pag/less_fix/frac_1 --exp_name cv1

bsub -o fra-cv.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--valid --test --method pet_attn --save_dir /home/karthikp/pag/less_fix/frac_2 --exp_name cv1

########################################################################################################################
