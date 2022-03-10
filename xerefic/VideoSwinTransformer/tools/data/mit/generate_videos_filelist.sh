#!/usr/bin/env bash

cd ../../../../../
PYTHONPATH=. python3 VideoSwinTransformer/tools/data/build_file_list.py mit data/videos/training --level 2 --format videos --num-split 1 --subset train --shuffle
echo "Train filelist for videos generated."

PYTHONPATH=. python3 VideoSwinTransformer/tools/data/build_file_list.py mit data/videos/training --level 2 --format videos --num-split 1 --subset val --shuffle
echo "Val filelist for videos generated."
cd tools/data/mit/
