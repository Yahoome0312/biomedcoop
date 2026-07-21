#!/bin/bash

# Reproduce BiomedCoOp on DermaMNIST across the complete few-shot sweep.
# [Reproduction addition] 32-shot is included explicitly; the upstream
# scripts only documented 1, 2, 4, 8, and 16 shots.
set -e

DATA=$1
MODEL=${2:-BiomedCLIP}

for SHOTS in 1 2 4 8 16 32
do
    bash scripts/biomedcoop/few_shot.sh "${DATA}" dermamnist "${SHOTS}" "${MODEL}"
done
