declare -a fractions=(0 1 2 3 4 5)

for j in "${fractions[@]}";
do python validate/cv_analyze.py --dir pag/fractions/frac_"$j"/;
done

for j in "${fractions[@]}";
do python validate/cv_analyze.py --dir pag/less/frac_"$j"/PAG-ct-pet;
done

for j in "${fractions[@]}";
do python validate/cv_analyze.py --dir pag/less/frac_"$j"/PAG-ct;
done

# python validate/rstr_analyze.py