
# Script for test run
# Test time

# ++++++++++++++++++++++++++++++++++ Baselines ++++++++++++++++++++++++++++++++++

bsub -o ts-bi.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u main.py \
--train --config configs/config_bi.json

bsub -o ts-bi.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u main.py \
--train --config configs/config_bi_attn.json

bsub -o ts-bi.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u main.py \
--train --config configs/config_uni.json

bsub -o ts-bi.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u main.py \
--train --config configs/config_uni_attn.json


# ++++++++++++++++++++++++++++++++++ PAG  ++++++++++++++++++++++++++++++++++
bsub -o ts-0.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--train --config configs/config_0.json --method pet_attn --save_dir experiments/pag/less/frac_0

bsub -o ts-1.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--train --config configs/config_1.json --method pet_attn --save_dir experiments/pag/less/frac_1

bsub -o ts-2.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--train --config configs/config_2.json --method pet_attn --save_dir experiments/pag/less/frac_2

bsub -o ts-3.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--train --config configs/config_3.json --method pet_attn --save_dir experiments/pag/less/frac_3

bsub -o ts-4.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--train --config configs/config_4.json --method pet_attn --save_dir experiments/pag/less/frac_4

bsub -o ts-5.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--train --config configs/config_5.json --method pet_attn --save_dir experiments/pag/less/frac_5

bsub -o ts-complete.txt -W 48:00 -R "rusage[mem=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=14000]" python -u loader.py \
--train --config configs/config_bi.json --method pet_attn --save_dir experiments/pag/less/complete