## code for generate MV and index for dataset


# # for GOPRO dataset
# IMG_DIR=/home/songzhuoran/video/video-sr-acc/GOPRO/BIx4
# OUT_DIR=/home/songzhuoran/video/video-sr-acc/GOPRO/Info_BIx4

# for GOPRO dataset
IMG_DIR=/home/songzhuoran/video/video-sr-acc/GOPRO/GT
OUT_DIR=/home/songzhuoran/video/video-sr-acc/GOPRO/Info_GT

video_gen(){
    mkdir -p $OUT_DIR/video
    mkdir -p $OUT_DIR/mvs

    rm -rf $OUT_DIR/video/*
    rm -rf $OUT_DIR/mvs/*

    for class_str in $(ls $IMG_DIR); do
        class=${class_str##*/}
        echo ${class}\'s video start producing ...
        ffmpeg -i $IMG_DIR/$class/%06d.png -vcodec libx265 -x265-params lossless=1 tmp.mp4
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


#################need to modify
video_gen
idx_gen

