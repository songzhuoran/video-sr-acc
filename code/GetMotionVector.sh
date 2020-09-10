## code for generate MV and index for dataset

# # for Vid4 dataset
# IMG_DIR=./Vid4/BIx4
# OUT_DIR=./Info_BIx4

# # for REDS dataset
# IMG_DIR=/home/songzhuoran/video/video-sr-acc/REDS/BIx4
# OUT_DIR=/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4

# for Vimeo90k dataset
IMG_DIR=/home/songzhuoran/video/video-sr-acc/Vimeo90K/BIx4
OUT_DIR=/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4

video_gen(){
    mkdir -p $OUT_DIR/video
    mkdir -p $OUT_DIR/mvs

    rm -rf $OUT_DIR/video/*
    rm -rf $OUT_DIR/mvs/*

    for class_str in $(ls $IMG_DIR); do
        class=${class_str##*/}
        echo ${class}\'s video start producing ...
        ffmpeg -i $IMG_DIR/$class/%08d.png -vcodec libx265 -x265-params lossless=1 tmp.mp4
        #./FFmpeg/ffmpeg -pattern_type glob -i '$IMG_DIR/$class/%08d.jpg' -vcodec libx265 -x265-params lossless=1 tmp.mp4
        ffmpeg -i tmp.mp4 -vcodec libx265 -x265-params lossless=1 $OUT_DIR/video/$class.mp4 > $OUT_DIR/mvs/$class.csv
        rm tmp.mp4
        echo ${class}\'s video done successfully. 
    done
    echo "Video & mvs generation step success."
}

idx_gen(){
    mkdir -p $OUT_DIR/idx/p
    mkdir -p $OUT_DIR/idx/b

    rm -rf $OUT_DIR/idx/p/*
    rm -rf $OUT_DIR/idx/b/*

    for class_str in $IMG_DIR/*; do
	class=${class_str##*/}
	ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv $OUT_DIR/video/$class.mp4 | grep -n [IP] | cut -d ':' -f 1 > $OUT_DIR/idx/p/$class
	ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv $OUT_DIR/video/$class.mp4 | grep -n [B] | cut -d ':' -f 1 > $OUT_DIR/idx/b/$class
    done
    echo "Index generation step success."
}

bframe_gen(){
    rm -rf $OUT_DIR/train/bframe/*
    
    for class_str in $IMG_DIR/*; do
        class=${class_str##*/}
	cd ../code
	python bframe_gen.py $class
	echo "$class generation done. "
    cd ../util
    done
}

#################need to modify
video_gen
idx_gen

