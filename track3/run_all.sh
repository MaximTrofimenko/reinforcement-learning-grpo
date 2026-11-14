#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export HF_HOME="$PWD/.hf-cache"
output_dir="${1:-rerun}"

python make_dataset.py
python -m unittest test_task.py
mkdir -p "$output_dir"
python evaluate.py --split dev --output "$output_dir/base_dev.json"
python evaluate.py --split final_test --output "$output_dir/base_final.json"

for seed in 20261114 20261115 20261116; do
  if [[ "$seed" == 20261114 ]]; then
    kl_dir="$output_dir"
    no_kl_dir="$output_dir/no_kl"
    kl_final="$output_dir/kl_final.json"
    no_kl_final="$output_dir/no_kl_final.json"
  else
    suffix="${seed: -2}"
    kl_dir="$output_dir/kl_seed$suffix"
    no_kl_dir="$output_dir/no_kl_seed$suffix"
    kl_final="$kl_dir/final.json"
    no_kl_final="$no_kl_dir/final.json"
  fi
  python train_grpo.py --steps 200 --eval-every 50 --kl-beta 0.02 --seed "$seed" --output "$kl_dir"
  python evaluate.py --split final_test --adapter "$kl_dir/adapter" --output "$kl_final"
  python train_grpo.py --steps 200 --eval-every 50 --kl-beta 0 --seed "$seed" --output "$no_kl_dir"
  python evaluate.py --split final_test --adapter "$no_kl_dir/adapter" --output "$no_kl_final"
done

python analyze.py --results-dir "$output_dir"
