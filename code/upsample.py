import sys
import os
import csv
import cv2
import numpy as np
import os
from PIL import Image
from scipy import interpolate


OUT_DIR = "/home/songzhuoran/video/video-sr-acc/Upsample/REDS/"
# OUT_DIR = "upsample/cubic/"
PICS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/BIx4/"



def interp_upsample2(classname):
    video_path = PICS_DIR + classname
    pic_names = os.listdir(video_path)
    frame_num = len(pic_names)
    img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
    frame_h, frame_w, _ = img.shape
    for i in range(frame_num):
        print('i: ', i)
        # img_org = cv2.imread(PICS_DIR + classname + '/' + pic_names[i])
        img_org = cv2.imread(PICS_DIR + classname + "/%08d.png" % i)
        # img_ext = cv2.resize(img_org, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        img_ext = cv2.resize(img_org, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        # print(OUT_DIR + classname + "/%08d.png" % i)
        cv2.imwrite(OUT_DIR + classname + "/%08d.png" % i, img_ext)

# for name in ['calendar', 'city', 'foliage', 'walk']:
for name in ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029']:
# for name in ['000','004']:
    interp_upsample2(name)