#!/usr/bin/env bash

# set up environment
source activate kinetics
pip install --upgrade youtube-dl
pip install tqdm

DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

DATA_DIR="/home/ubuntu/ModelExtraction/data/$DATASET/"
ANNO_DIR="/home/ubuntu/ModelExtraction/data/$DATASET/annotations"
# python3 download.py ${ANNO_DIR}/kinetics_train.csv ${DATA_DIR}/training
python3 download.py ${ANNO_DIR}/kinetics_val.csv ${DATA_DIR}/validation

source deactivate kinetics
