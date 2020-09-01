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
TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/Vid4_Cluster/"


overall_info = [] # a list to store all info, including MV and frequency
# classname_list = ['calendar','city','foliage','walk']
classname_list = ['walk'] # need to modify!!!!
MV_list = []
res_list = []
# iterate all videos

### code for clustering 8*8 residual blocks
def Cluster_res_func(ratio,interval):
    overall_info = [] # a list to store all info, including MV and frequency
    MV_list = []
    res_list = []
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
                
            if (frame_cnt % interval ==0) and (frame_cnt != 0):
                res_arr = np.array(res_list)

                # ## MeanShift clustering algorithm
                # ms = MeanShift(bin_seeding=True,n_jobs=32)
                # cluster_label = ms.fit_predict(res_arr)

                ## K-means clustering algorithm
                ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio))
                cluster_label = ms.fit_predict(res_arr)

                ## calculate the clustered residual
                clustered_res_list = []
                clustered_res_list = list(ms.cluster_centers_.astype("float"))

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

            # ## MeanShift clustering algorithm
            # ms = MeanShift(bin_seeding=True,n_jobs=32)
            # cluster_label = ms.fit_predict(res_arr)

            ## K-means clustering algorithm
            ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio))
            cluster_label = ms.fit_predict(res_arr)
            
            ## calculate the clustered residual
            clustered_res_list = []
            clustered_res_list = list(ms.cluster_centers_.astype("float"))

            overall_cluster_label.append(cluster_label)
            overall_clusered_res.append(clustered_res_list)
            MV_list = []
            res_list = []
            frame_cnt = 0
            print("final succeed!")

        # ## MeanShift clustering algorithm
        # cluster = shelve.open(TRAIN_DIR+"Mean_shift_label_i"+str(interval)+".bat")
        # cluster[classname] = overall_cluster_label
        # cluster.close()

        # cluster_res = shelve.open(TRAIN_DIR+"Mean_shift_res_i"+str(interval)+".bat")
        # cluster_res[classname] = overall_clusered_res
        # cluster_res.close()

        ## K-means algorithm
        cluster = shelve.open(TRAIN_DIR+"cluster_label_r"+str(ratio)+"_i"+str(interval)+".bat")
        cluster[classname] = overall_cluster_label
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+"cluster_res_r"+str(ratio)+"_i"+str(interval)+".bat")
        cluster_res[classname] = overall_clusered_res
        cluster_res.close()
    return

### code for clustering clustered residual
def Cluster_clustered_res_func(ratio_clustered,ratio,interval):
    # print(ratio_clustered,ratio,interval)
    for i in classname_list:
        classname = i
        print("classname: ",classname)
        cluster = shelve.open(TRAIN_DIR+"cluster_label_r"+str(ratio)+"_i"+str(interval)+".bat")
        overall_cluster_label = cluster[classname]
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"cluster_res_r"+str(ratio)+"_i"+str(interval)+".bat")
        overall_clusered_res = cluster_res[classname]
        cluster_res.close()

        overall_cluster_clustered_label = []
        overall_cluster_clusered_res = []

        for label_cnt in range(len(overall_cluster_label)):
            cluster_label = overall_cluster_label[label_cnt]
            clustered_res = overall_clusered_res[label_cnt]
            # print(clustered_res[0].shape)
            res_list = []
            for label_idx, label_data in enumerate(cluster_label):
                tmp_res = clustered_res[label_data]
                res_list.append(tmp_res)
            #apply K-means to cluster clustered residual
            res_arr = np.array(res_list)
            ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio_clustered))
            cluster_clustered_label = ms.fit_predict(res_arr)
            ## calculate the clustered residual

            # cluster_clustered_res_list = []
            # for label_num in range(int(res_arr.shape[0]/ratio_clustered)): # 0~1600
            #     cluster_clustered_res = np.zeros((1,192))
            #     clustered_cnt = 0
            #     for label_idx, label_data in enumerate(cluster_clustered_label):
            #         if label_data == label_num:
            #             clustered_cnt +=1
            #             cluster_clustered_res += res_arr[label_idx]
            #     # print("clustered_res: ", clustered_res)
            #     cluster_clustered_res_list.append((cluster_clustered_res.astype("float")/float(clustered_cnt)).astype("float"))
            #     clustered_cnt = 0
            clustered_res_list = []
            clustered_res_list = list(ms.cluster_centers_.astype("float"))

            overall_cluster_clustered_label.append(cluster_clustered_label)
            overall_cluster_clusered_res.append(cluster_clustered_res_list)
        
        #store back
        cluster_tmp = shelve.open(TRAIN_DIR+"cluster_clustered_label_r"+str(ratio_clustered)+"_i"+str(interval)+".bat")
        cluster_tmp[classname] = overall_cluster_clustered_label
        cluster_tmp.close()

        cluster_res_tmp = shelve.open(TRAIN_DIR+"cluster_clustered_res_r"+str(ratio_clustered)+"_i"+str(interval)+".bat")
        cluster_res_tmp[classname] = overall_cluster_clusered_res
        cluster_res_tmp.close()
    return

### code for clustering delta residual = (residual-clustered residual)
def Cluster_delta_func(ratio_delta,ratio,interval):
    for i in classname_list:
        classname = i
        print("classname: ",classname)
        cluster = shelve.open(TRAIN_DIR+"cluster_label_r"+str(ratio)+"_i"+str(interval)+".bat")
        overall_cluster_label = cluster[classname] # clustered label
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"cluster_res_r"+str(ratio)+"_i"+str(interval)+".bat")
        overall_clusered_res = cluster_res[classname] # central residual
        cluster_res.close()
        she = shelve.open(TRAIN_DIR+"residual.bat") # all residual for 8*8 blocks
        overall_info = she[classname]
        she.close()

        block_MV_list = []
        block_res_list = []
        for block_info in overall_info:
            ## obtain 8*8 MV and residual
            block_MV = block_info[0].reshape(-1)
            block_res = block_info[1].reshape(-1)
            cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = block_MV
            block_MV_list.append(block_MV) #append MV into list
            block_res_list.append(block_res)

        begin_cnt = 0 # use to locate MV in block_MV_list and block_res_list
        overall_delta_res_list = [] # use to store delta residual

        # calculate the delta of central residual and residual
        for label_cnt in range(len(overall_cluster_label)):
            delta_res_list = []
            cluster_label = overall_cluster_label[label_cnt]
            clustered_res = overall_clusered_res[label_cnt]
            for label_idx, label_data in enumerate(cluster_label): # 0~16000 item number
                # locate MV
                MV_idx = label_idx + begin_cnt
                tmp_MV = block_MV_list[MV_idx]
                tmp_res = block_res_list[MV_idx]

                # locate res
                single_res = clustered_res[label_data] #shape = (1,192)
                delta_res = tmp_res - single_res
                delta_res_list.append(delta_res)
            overall_delta_res_list.append(delta_res_list) # a list storing the delta of central residual and residual
            begin_cnt += len(overall_cluster_label[label_cnt])

        
        # clustering the delta of central residual and residual, then calculate the central delta residual
        overall_clustered_delta_label_list = []
        overall_clustered_delta_res_list = []
        for label_cnt in range(len(overall_delta_res_list)):
            delta_res_arr = np.array(overall_delta_res_list[label_cnt])
            print(delta_res_arr.shape)
            if ratio_delta !=0:
                ms = KMeans(n_clusters=int(delta_res_arr.shape[0]/ratio_delta))
                delta_cluster_label = ms.fit_predict(delta_res_arr)

                clustered_delta_res_list = []
                clustered_delta_res_list = list(ms.cluster_centers_.astype("float"))                     

                overall_clustered_delta_label_list.append(delta_cluster_label)
                overall_clustered_delta_res_list.append(clustered_delta_res_list)
            
            else: # used for test whether the code flow is correct
                delta_cluster_label = np.zeros((delta_res_arr.shape[0]))
                for dd_cc_idx in range(int(delta_cluster_label.shape[0])):
                    delta_cluster_label[dd_cc_idx] = int(dd_cc_idx)
                clustered_delta_res_list = []
                for label_num in range(int(delta_cluster_label.shape[0])): # 0~405 label number
                    delta_clustered_res = np.zeros((1,192))
                    delta_clustered_res = delta_res_arr[label_num]
                    clustered_delta_res_list.append(delta_clustered_res.astype("float"))
                    clustered_cnt = 0                

                overall_clustered_delta_label_list.append(delta_cluster_label)
                overall_clustered_delta_res_list.append(clustered_delta_res_list)

        cluster = shelve.open(TRAIN_DIR+"cluster_delta_label_r"+str(ratio_delta)+"_i"+str(interval)+".bat")
        cluster[classname] = overall_clustered_delta_label_list
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+"cluster_delta_res_r"+str(ratio_delta)+"_i"+str(interval)+".bat")
        cluster_res[classname] = overall_clustered_delta_res_list
        cluster_res.close()
        print("success!")
    return

# if __name__ == '__main__':
ratio = int(sys.argv[1])
interval = int(sys.argv[2])
Cluster_res_func(ratio,interval)
# compression_ratio = 16 # compress weights 16x

# # ratio_clustered = int(compression_ratio/ratio)
# # Cluster_clustered_res_func(ratio_clustered,ratio,interval)

# ratio_delta = int(compression_ratio*ratio/(ratio-compression_ratio))
# Cluster_delta_func(ratio_delta,ratio,interval)


