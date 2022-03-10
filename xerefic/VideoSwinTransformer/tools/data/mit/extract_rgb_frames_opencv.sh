#!/usr/bin/env bash

#cd ../
python3 xerefic/VideoSwinTransformer/tools/data/build_rawframes.py /home/ubuntu/ModelExtraction/data/extracted/training /home/ubuntu/ModelExtraction/data/rawframes/training --level 2 --ext mp4 --task rgb --use-opencv
echo "Raw frames (RGB only) generated for train set"

#python xerefic/VideoSwinTransformer/tools/data/build_rawframes.py /home/ubuntu/ModelExtraction/data/extracted/validation/ /home/ubuntu/ModelExtraction/data/rawframes/validation --level 2 --ext mp4 --task rgb --use-opencv
#echo "Raw frames (RGB only) generated for val set"

#cd mit/
