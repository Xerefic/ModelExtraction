#wget http://data.csail.mit.edu/soundnet/actions3/split2/Moments_in_Time_Raw_v2.zip
#unzip Moments_in_Time_Raw_v2.zip -d data/
#mv data/Moments_in_Time_Raw data/videos
#rm -rf Moments_in_Time_Raw_v2.zip

#python3 xerefic/VideoSwinTransformer/tools/data/resize_video.py data/videos/training data/extracted/training --dense --level 2
#rm -rf data/videos/

./xerefic/VideoSwinTransformer/tools/data/mit/extract_rgb_frames_opencv.sh
