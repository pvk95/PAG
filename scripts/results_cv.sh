declare -a fractions=(0 1 2 3 4 5)

echo "++++++++++++++++++++++++++++++++++++++ Generating metrics +++++++++++++++++++++++++++++++++"
python validate/cv_analyze.py --dir multi-baselines/ct/ --gen_cv_metrics

python validate/cv_analyze.py --dir multi-baselines/ct_attn/ --gen_cv_metrics

python validate/cv_analyze.py --dir pag/less/complete/PAG-ct/ --gen_cv_metrics

python validate/cv_analyze.py --dir pag/less/complete/PAG-ct-pet/ --gen_cv_metrics

python validate/cv_analyze.py --dir pag/fractions/complete/ --gen_cv_metrics

python validate/cv_analyze.py --dir multi-baselines/bimodal_attn/ --gen_cv_metrics


for j in "${fractions[@]}";
do python validate/cv_analyze.py --dir pag/less/frac_"$j"/PAG-ct-pet --gen_cv_metrics;
done


for j in "${fractions[@]}";
do python validate/cv_analyze.py --dir pag/less/frac_"$j"/PAG-ct --gen_cv_metrics;
done

for j in "${fractions[@]}";
do python validate/cv_analyze.py --dir pag/fractions/frac_"$j"/ --gen_cv_metrics;
done


##################################################################################################
# ++++++++++++++++++++++++++++++++++ Perform dir_nalyze +++++++++++++++++++++++++++++++++++++++++++
echo "++++++++++++++++++++++++++++++++++++ Performing dir_analyze +++++++++++++++++++++++++++++++++"
python validate/dir_analyze.py --dirs \
multi-baselines/ct/ \
multi-baselines/ct_attn/ \
pag/less/complete/PAG-ct/ \
pag/less/complete/PAG-ct-pet/ \
pag/fractions/complete/ \
multi-baselines/bimodal_attn/ --save_dir metrics_plots_v1/

####################################################################################################
# ++++++++++++++++++++++++++++++++++ Perform fractions analyze ++++++++++++++++++++++++++++++++++++++

echo "++++++++++++++++++++++++++++++++++++ Performing fractions analyze +++++++++++++++++++++++++++++++++"
python validate/rstr_analyze.py --save_dir metrics_plots_v1/