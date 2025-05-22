#!/bin/bash
set -e

subjects=(107422 117930 149236 164030 168341 519950 613538 734045 751348 114621)
TRAIN_SPLITS=(0.1 0.3 0.5 0.7)
BASE_CONFIG="cfg/wiren_hcp_bvectors.yml"

export BASE_CONFIG

parallel -j 16 \
    'sub={1}; split={2}; \
    RESULTS_DIR="EVALUATIONS/bvecs_results_norm/${sub}/split_${split}"; \
    mkdir -p "$RESULTS_DIR"; \
    TMP_CONFIG="tmp_config_${sub}_${split}.yml"; \
    sed -e "s/train_split: [0-9]*\.[0-9]*/train_split: $split/" \
        "$BASE_CONFIG" > "$TMP_CONFIG"; \
    python ../../src/evaluation_shore.py \
           "$TMP_CONFIG" \
           "$RESULTS_DIR" \
           --subject "$sub" \
           --no_denorm;' ::: "${subjects[@]}" ::: "${TRAIN_SPLITS[@]}"

echo "All subjects and splits processed successfully!"
