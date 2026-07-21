#!/bin/bash

# custom config
DATA=$1
DATASET=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16, 32)
# [Reproduction addition] 32-shot was not listed in the original script;
# the generic dataset sampler supports it, so it is explicitly accepted here.
case "${SHOTS}" in
    1|2|4|8|16|32) ;;
    *) echo "Unsupported shot count: ${SHOTS}; use 1, 2, 4, 8, 16, or 32"; exit 2 ;;
esac
MODEL=$4
NCTX=4
CSC=False
CTP=end

METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}

for SEED in 1 2 3
do
        DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
           python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
            --output-dir ${DIR} \
            TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
            TRAINER.BIOMEDCOOP.CSC ${CSC} \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
done
