declare -a folds=(0 1 2 3)
declare -a fractions=(0 1 2 3 4 5)

# ++++++++++++++++++++++++++++++++++ Unimodal ++++++++++++++++++++++++++++++++++
for i in "${folds[@]}";
do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python -u validate/validate.py --dirs multi-baselines/ct/cv"$i"/ \
--detect --analyze --valid --test;
done
bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs multi-baselines/ct/ts/ \
--detect --analyze --test;

# ++++++++++++++++++++++++++++++++++ Unimodal_attn ++++++++++++++++++++++++++++++++++
for i in "${folds[@]}";
do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs multi-baselines/ct_attn/cv"$i"/ \
--detect --analyze --valid --test;
done
bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs multi-baselines/ct_attn/ts/ \
--detect --analyze --test;

# ++++++++++++++++++++++++++++++++++ Bimodal ++++++++++++++++++++++++++++++++++
for i in "${folds[@]}";
do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/fractions/complete/cv"$i"/ \
--detect --analyze --valid --test;
done
bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/fractions/complete/ts/ \
--detect --analyze --test;

# ++++++++++++++++++++++++++++++++++ Bimodal_attn ++++++++++++++++++++++++++++++++++
for i in "${folds[@]}";
do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs multi-baselines/bimodal_attn/cv"$i"/ \
--detect --analyze --valid --test;
done
bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs multi-baselines/bimodal_attn/ts/ \
--detect --analyze --test;

# ++++++++++++++++++++++++++++++++++ PAG-ct ++++++++++++++++++++++++++++++++++
for i in "${folds[@]}";
do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/complete/PAG-ct/cv"$i"/ \
--detect --analyze --valid --test;
done
bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/complete/PAG-ct/ts/ \
--detect --analyze --test;


# ++++++++++++++++++++++++++++++++++ PAG-ct-pet ++++++++++++++++++++++++++++++++++
for i in "${folds[@]}";
do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/complete/PAG-ct-pet/cv"$i"/ \
--detect --analyze --valid --test;
done
bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/complete/PAG-ct-pet/ts/ \
--detect --analyze --test;

declare -a folds=(0 1 2 3)
declare -a fractions=(0 1 2 3 4 5)

###########################################################################################
# ++++++++++++++++++++++++++++++++++ Fractions PAG-ct-pet ++++++++++++++++++++++++++++++++++

for j in "${fractions[@]}";

do
  for i in "${folds[@]}";
    do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/frac_"$j"/PAG-ct-pet/cv"$i"/ \
    --detect --analyze --valid --test;
  done
  bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/frac_"$j"/PAG-ct-pet/ts/ --detect --analyze --test;
done

###########################################################################################
# ++++++++++++++++++++++++++++++++++ Fractions PAG-ct ++++++++++++++++++++++++++++++++++++++

for j in "${fractions[@]}";
do
  for i in "${folds[@]}";
    do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/frac_"$j"/PAG-ct/cv"$i"/ \
    --detect --analyze --valid --test;
  done
  bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/less/frac_"$j"/PAG-ct/ts/ --detect --analyze --test;
done

###########################################################################################
# ++++++++++++++++++++++++++++++++++ Fractions Bimodal ++++++++++++++++++++++++++++++++++++++

for j in "${fractions[@]}";
do
  for i in "${folds[@]}";
  do bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/fractions/frac_"$j"/cv"$i"/ \
  --detect --analyze --valid --test;
  done
  bsub -o val.txt -n 1 -R "rusage[mem=15000]" python validate/validate.py --dirs pag/fractions/frac_"$j"/ts/ --detect --analyze --test;
done


