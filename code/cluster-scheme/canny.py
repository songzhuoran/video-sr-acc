import csv
import os
import re
import cv2
import numpy as np
import math

Video_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/GT/"
Out_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/canny/"


video_names = os.listdir(Video_DIR)
for video_name in video_names:
    # video_name="blackswan"
    print(video_name)


    img_names=os.listdir(Video_DIR+video_name)
    for img_name in img_names:
        img1 = cv2.imread(Video_DIR+video_name+"/"+img_name,0)
        img2=cv2.Canny(img1,150,250) # image after canny
        cv2.imwrite(Out_DIR+video_name+"/"+img_name,img2)


        


