#!/bin/bash

# Check if subject ID is provided
if [ -z "$1" ]; then
    echo "Please provide a subject ID"
    echo "Usage: ./run_all_splits.sh <subject_id>"
    exit 1
fi

SUBJECT_ID=$1
TRAIN_SPLITS=(0.1 0.3 0.5 0.7)
#TRAIN_SPLITS=(0.7)
# Create a temporary config file
TMP_CONFIG="tmp_config.yml"

for split in "${TRAIN_SPLITS[@]}"; do
    echo "Running training with train_split=$split for subject $SUBJECT_ID"
    
    # Create a temporary config file with the current split
    sed -e "s/train_split: 0.1/train_split: $split/" \
        -e "s#checkpoint_dirpath: \"OUTPUTS/sample_bvectors/subject/split_0.1\"#checkpoint_dirpath: \"OUTPUTS/sample_bvectors2/subject/split_$split\"#" \
        experiments/1_exp_no_bvecs/cfg/wiren_hcp_bvectors.yml > "$TMP_CONFIG"
    
    # Run the training
    python src/main.py "$TMP_CONFIG" --subject "$SUBJECT_ID"
    
    # Clean up temporary config
    rm "$TMP_CONFIG"
    
    echo "Completed training for split $split"
    echo "----------------------------------------"
done

echo "All training runs completed!" 