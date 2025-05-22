#!/bin/bash
set -e

MAIN=../../src/main.py
CFG=cfg/wiren_template.yml

python "$MAIN" "$CFG"
