#!/bin/bash
set -e

# Define the array of subjects
# subjects=(107422 117930 149236 164030 168341 519950 613538 734045 751348 114621)
subjects=(114621)  # Uncomment to test with single subject

# Define the training splits to evaluate
TRAIN_SPLITS=(0.1 0.3 0.5 0.7)

# Create a temporary config file
TMP_CONFIG="tmp_config.yml"

for sub in "${subjects[@]}"; do
    echo "Processing subject $sub"
    
    for split in "${TRAIN_SPLITS[@]}"; do
        echo "Evaluating model with train_split=$split for subject $sub"
        
        # Create results directory with split info
        RESULTS_DIR="OUTPUTS/bvecs_results/${sub}/split_${split}"
        mkdir -p "$RESULTS_DIR"
        
        # Run the evaluation with the specific split model
        python ../../src/evaluation_shore.py \
               cfg/wiren_hcp_bvectors.yml \
               "$RESULTS_DIR" \
               --subject "$sub" \
               --train_split "$split"
               
        echo "Completed evaluation for subject $sub with split $split"
        echo "----------------------------------------"
    done
    
    echo "All evaluations for subject $sub completed!"
    echo "========================================"
done

echo "All subjects and splits processed successfully!"
