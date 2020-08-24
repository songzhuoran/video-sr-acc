### code for reconstructing 8*8 residual blocks
import warnings
warnings.filterwarnings("ignore")
import sys
import os
import csv
import cv2
import numpy as np
import os
from PIL import Image
from scipy import interpolate
import shelve
from sklearn.cluster import *

#directory
B_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/bframe_sr_reconstruction/" # SR result
TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/"
PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/BIx4/" # GT_LR_pic


overall_info = [] # a list to store all info, including MV and frequency
# classname_list = ['calendar','city','foliage','walk']
classname_list = ['calendar'] # need to modify!!!!!
MV_list = []
res_list = []
# iterate all videos


for i in classname_list:
    classname = i
    print("classname: ",classname)
    cluster = shelve.open(TRAIN_DIR+"cluster_label.bat")
    overall_cluster_label = cluster[classname]
    cluster.close()
    cluster_res = shelve.open(TRAIN_DIR+"cluster_res.bat")
    overall_clusered_res = cluster_res[classname]
    cluster_res.close()
    she = shelve.open(TRAIN_DIR+"residual.bat") # for Vid4 dataset
    overall_info = she[classname]
    she.close()

    #init frame info
    video_path = PICS_DIR + classname
    pic_names = os.listdir(video_path)
    frame_num = len(pic_names)
    img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = img.shape
    sr_frame_h = 4 * frame_h
    sr_frame_w = 4 * frame_w
    frame_mat_SR = np.zeros((frame_num,sr_frame_h,sr_frame_w, 3), dtype="uint8") #init frame_mat
    for i in range(frame_num):
        # print(B_DIR + classname + "/%08d.png" % i)
        tmp_img = cv2.imread(B_DIR + classname + "/%08d.png" % i) # read SR result
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        frame_mat_SR[i] = tmp_img


    block_MV_list = []
    # print(len(overall_cluster_label))
    for block_info in overall_info:
        ## obtain 8*8 MV and residual
        block_MV = block_info[0].reshape(-1)
        block_res = block_info[1].reshape(-1)
        cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = block_MV
        block_MV_list.append(block_MV)

    begin_cnt = 0 # use to locate MV in block_MV_list
    tmp_cur_idx = -1
    for label_cnt in range(len(overall_cluster_label)):
        print(label_cnt)
        cluster_label = overall_cluster_label[label_cnt]
        clustered_res = overall_clusered_res[label_cnt]
        for label_idx, label_data in enumerate(cluster_label):
            # locate MV
            MV_idx = label_idx + begin_cnt
            tmp_MV = block_MV_list[MV_idx]
            cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = tmp_MV

            #read and write a B frame
            if tmp_cur_idx == -1:
                print("init a B frame, the idx is %d: ", cur_idx)
                B_img = cv2.imread(B_DIR + classname + '/' + "%08d.png" % cur_idx, -1)
                B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB).astype("float")
            elif tmp_cur_idx != cur_idx:
                print("write block a B frame, the idx is %d: ",tmp_cur_idx)
                # boundary condition
                for i in range(sr_frame_w):
                    for j in range(sr_frame_h):
                        for c in range(3):
                            if B_img[j,i,c]<0:
                                B_img[j,i,c] = 0
                            if B_img[j,i,c]>255:
                                B_img[j,i,c] = 255
                B_img = B_img.astype("uint8")
                B_img = cv2.cvtColor(B_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(B_DIR + classname + '/' + "%08d.png" % tmp_cur_idx, B_img)
                print("init a B frame, the idx is %d: ", cur_idx)
                B_img = cv2.imread(B_DIR + classname + '/' + "%08d.png" % cur_idx, -1)
                B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB).astype("float")
            tmp_cur_idx = cur_idx

            # locate res
            tmp_res = clustered_res[label_data] #shape = (1,192)
            tmp_res = tmp_res.reshape((block_h,block_w,3))

            #reconstruction
            ref_frame_sr = frame_mat_SR[ref_idx]
            for px in range(block_w):
                for py in range(block_h):
                    if ((curx+px) < sr_frame_w) and ((cury+py) < sr_frame_h):
                        if ((refx + px) < sr_frame_w) and ((refy+py) < sr_frame_h):
                            B_img[cury+py,curx+px,:] = ref_frame_sr[refy+py,refx + px,:].astype("float") + tmp_res[py,px,:].astype("float")
                        else:
                            B_img[cury+py,curx+px,:] = tmp_res[py,px,:].astype("float")

            
        begin_cnt += len(overall_cluster_label[label_cnt])
        


    



    

