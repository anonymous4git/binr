#!/bin/bash
set -e

MAIN=../../src/evaluation.py
CFG=./cfg/wiren_hcp.yml
OUT=./results/exp0/hcp

python "$MAIN" "$CFG" --no_denorm --subject 107422 "$OUT/107422"
python "$MAIN" "$CFG" --no_denorm --subject 117930 "$OUT/117930"
python "$MAIN" "$CFG" --no_denorm --subject 149236 "$OUT/149236"
python "$MAIN" "$CFG" --no_denorm --subject 164030 "$OUT/164030"
python "$MAIN" "$CFG" --no_denorm --subject 168341 "$OUT/168341"
python "$MAIN" "$CFG" --no_denorm --subject 519950 "$OUT/519950"
python "$MAIN" "$CFG" --no_denorm --subject 613538 "$OUT/613538"
python "$MAIN" "$CFG" --no_denorm --subject 734045 "$OUT/734045"
python "$MAIN" "$CFG" --no_denorm --subject 751348 "$OUT/751348"
python "$MAIN" "$CFG" --no_denorm --subject 114621 "$OUT/114621"

