#!/bin/bash
set -e

# Define the array of subjects
subjects=(107422 117930 149236 164030 168341 519950 613538 734045 751348 114621)
#subjects=(117930)  # Uncomment to test with single subject

# Define the training splits to evaluate
TRAIN_SPLITS=(0.1 0.3 0.5 0.7)

# Base config file
BASE_CONFIG="experiments/1_exp_no_bvecs/cfg/wiren_hcp_bvectors.yml"

for sub in "${subjects[@]}"; do
    echo "Processing subject $sub"
    
    for split in "${TRAIN_SPLITS[@]}"; do
        echo "Evaluating model with train_split=$split for subject $sub"
        
        # Create results directory with split info
        RESULTS_DIR="EVALUATIONS/bvecs_results_norm/${sub}/split_${split}"
        mkdir -p "$RESULTS_DIR"
        
        # Create a temporary config file with the current split
        TMP_CONFIG="tmp_config_${sub}_${split}.yml"
        sed -e "s/train_split: [0-9]*\.[0-9]*/train_split: $split/" \
            -e "s#checkpoint_dirpath: \"OUTPUTS/sample_bvectors/subject/split_[0-9]*\.[0-9]*\"#checkpoint_dirpath: \"OUTPUTS/sample_bvectors/${sub}/split_$split\"#" \
            -e "s#evaluation_checkpoint: \"OUTPUTS/sample_bvectors/subject/last.ckpt\"#evaluation_checkpoint: \"OUTPUTS/sample_bvectors/${sub}/split_$split/last.ckpt\"#" \
            "$BASE_CONFIG" > "$TMP_CONFIG"
        
        # Run the evaluation with the specific split model via config file
        python src/evaluation.py \
               "$TMP_CONFIG" \
               "$RESULTS_DIR" \
               --subject "$sub" \
               --no_denorm
               
        # Clean up temporary config
        rm "$TMP_CONFIG"
        
        echo "Completed evaluation for subject $sub with split $split"
        echo "----------------------------------------"
    done
    
    echo "All evaluations for subject $sub completed!"
    echo "========================================"
done

echo "All subjects and splits processed successfully!"