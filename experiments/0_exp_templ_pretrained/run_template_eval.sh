#!/bin/bash
set -e

MAIN=../../src/evaluation.py
CFG=cfg/wiren_template.yml
OUT=./results/exp0/template

python "$MAIN" "$CFG" "$OUT"
