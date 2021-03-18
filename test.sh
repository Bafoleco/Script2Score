#!/bin/sh

# mkdir test_out
# python3 main.py --rev --freq --cat --num > test_out/all_rev
# python3 main.py --freq --cat --num > test_out/all_score

# python3 main.py --rev --cat --num > test_out/cat_rev
# python3 main.py --cat --num > test_out/cat_score

python3 main.py --rev --freq > test_out/freq_rev_real
python3 main.py --freq > test_out/freq_score_real
