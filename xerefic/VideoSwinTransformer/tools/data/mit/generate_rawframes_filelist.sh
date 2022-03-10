#!/usr/bin/env bash

cd ../../../../../
PYTHONPATH=. python3 VideoSwinTransformer/tools/data/build_file_list.py mit data/rawframes/training --level 2 --format rawframes --num-split 1 --subset train --shuffle
echo "Train filelist for rawframes generated."

PYTHONPATH=. python3 VideoSwinTransformer/tools/data/build_file_list.py mit data/rawframes/validation --level 2 --format rawframes --num-split 1 --subset val --shuffle
echo "Val filelist for rawframes generated."
cd tools/data/mit/
