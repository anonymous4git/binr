#!/bin/bash

subjects=(107422 117930 149236 164030 168341 519950 613538 734045 751348 114621)

for sub in "${subjects[@]}"; do
    echo "Processing subject $sub"
    python experiments/3_md_fa_evaluation/compute_md_fa.py --results_dir "OUTPUTS/unregistered_results/$sub"
    
    
done
