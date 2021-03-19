#!/bin/sh

mkdir test_out_2
python3 main.py --rev --freq --cat --num > test_out_2/all_rev
python3 main.py --freq --cat --num > test_out_2/all_score

python3 main.py --rev --cat --num > test_out_2/cat_rev
python3 main.py --cat --num > test_out_2/cat_score

python3 main.py --rev --freq > test_out_2/freq_rev_real
python3 main.py --freq > test_out_2/freq_score_real

python3 main.py --rev --freq --cat --num --dir > test_out_2/all_rev+x
python3 main.py --freq --cat --num --dir > test_out_2/all_score+x
