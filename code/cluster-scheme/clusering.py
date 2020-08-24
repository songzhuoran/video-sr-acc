### code for clustering 8*8 residual blocks
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
TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/"


overall_info = [] # a list to store all info, including MV and frequency
# classname_list = ['calendar','city','foliage','walk']
classname_list = ['calendar'] # need to modify!!!!!
MV_list = []
res_list = []
# iterate all videos
for i in classname_list:
    classname = i
    print("classname: ",classname)
    she = shelve.open(TRAIN_DIR+"residual.bat") # for Vid4 dataset
    overall_info = she[classname] # residual
    she.close()
    tmp_frame = -1
    frame_cnt = 0
    overall_cluster_label = []
    overall_clusered_res = []
    for block_info in overall_info:
        ## obtain 8*8 MV and residual
        block_MV = block_info[0].reshape(-1)
        block_res = block_info[1].reshape(-1)
        cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = block_MV
        # print(cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy)
        if cur_idx!=tmp_frame:
            if tmp_frame != -1:
                frame_cnt += 1
            tmp_frame = cur_idx
            
        if (frame_cnt % 2 ==0) and (frame_cnt != 0):
            res_arr = np.array(res_list)

            ## K-means clustering algorithm
            ms = KMeans(n_clusters=int(res_arr.shape[0]/16))
            cluster_label = ms.fit_predict(res_arr)
            ## calculate the clustered residual
            clustered_res_list = []
            for label_num in range(int(res_arr.shape[0]/16)): # 0~1600
                clustered_res = np.zeros((1,192))
                clustered_cnt = 0
                for label_idx, label_data in enumerate(cluster_label):
                    if label_data == label_num:
                        clustered_cnt +=1
                        clustered_res += res_arr[label_idx]
                # print("clustered_res: ", clustered_res)
                clustered_res_list.append((clustered_res.astype("float")/float(clustered_cnt)).astype("float"))
                # print("res: ")
                # print((clustered_res.astype("float")/float(clustered_cnt)).astype("float"))
                clustered_cnt = 0
                

            overall_cluster_label.append(cluster_label)
            overall_clusered_res.append(clustered_res_list)
            MV_list = []
            res_list = []
            frame_cnt = 0
            print("succeed!")

        MV_list.append(block_MV)
        res_list.append(block_res)

    if len(res_list) != 0:
        res_arr = np.array(res_list)

        ## MeanShift clustering algorithm
        # ms = MeanShift(bin_seeding=True)
        # cluster_label = ms.fit_predict(res_arr)

        ## K-means clustering algorithm
        ms = KMeans(n_clusters=int(res_arr.shape[0]/16))
        cluster_label = ms.fit_predict(res_arr)
        ## calculate the clustered residual
        clustered_res_list = []
        for label_num in range(int(res_arr.shape[0]/16)): # 0~405
            clustered_res = np.zeros((1,192))
            clustered_cnt = 0
            for cluster_l in cluster_label:
                if cluster_l == label_num:
                    clustered_cnt +=1
                    clustered_res += res_arr[cluster_l]
            # print("clustered_res: ", clustered_res)
            clustered_res_list.append((clustered_res.astype("float")/float(clustered_cnt)).astype("float"))
            print("res: ")
            print((clustered_res.astype("float")/float(clustered_cnt)).astype("float"))
            clustered_cnt = 0

        overall_cluster_label.append(cluster_label)
        overall_clusered_res.append(clustered_res_list)
        MV_list = []
        res_list = []
        frame_cnt = 0
        print("final succeed!")


    cluster = shelve.open(TRAIN_DIR+"cluster_label.bat")
    cluster[classname] = overall_cluster_label
    cluster.close()

    cluster_res = shelve.open(TRAIN_DIR+"cluster_res.bat")
    cluster_res[classname] = overall_clusered_res
    cluster_res.close()



    

