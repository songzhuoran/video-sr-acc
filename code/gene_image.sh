## code for generate MV and index for dataset

# for Vid4 dataset
IMG_DIR=/home/songzhuoran/video/video-sr-acc/Vid4/Info_GT_test/video
OUT_DIR=/home/songzhuoran/video/video-sr-acc/Vid4/Info_GT_test/images


video_gen(){

    for class_str in $(ls $IMG_DIR); do
        class=${class_str%\.*}
        echo $class
        ffmpeg -i $IMG_DIR/$class_str -f image2 $OUT_DIR/$class/%08d.png
    done
}


#################need to modify
video_gen


