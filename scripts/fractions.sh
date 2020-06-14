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