#!/bin/bash
set -e

MAIN=../../src/main.py
CFG=./cfg/wiren_hcp_pre_warped.yml

python "$MAIN" "$CFG" --subject 107422 
python "$MAIN" "$CFG" --subject 117930 
python "$MAIN" "$CFG" --subject 149236 
python "$MAIN" "$CFG" --subject 164030 
python "$MAIN" "$CFG" --subject 168341 
python "$MAIN" "$CFG" --subject 519950 
python "$MAIN" "$CFG" --subject 613538 
python "$MAIN" "$CFG" --subject 734045 
python "$MAIN" "$CFG" --subject 751348 
python "$MAIN" "$CFG" --subject 114621 

