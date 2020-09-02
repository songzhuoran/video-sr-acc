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
IDX_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/idx/" # idx directory
overall_info = [] # a list to store all info, including MV and frequency
classname_list = ['calendar','city','foliage','walk']
# classname_list = ['walk'] # need to modify!!!!
MV_list = []
res_list = []


### code for clustering 8*8 residual blocks
def Cluster_res_func(ratio):

    MV_list = []
    res_list = []

    for i in classname_list:
        classname = i
        print("classname: ",classname)
        pflist = [] # a list to store the index of P frames

        ## load idx
        with open(IDX_DIR+"p/"+classname, "r") as file:
            for row in file:
                pflist.append(int(row)-1)
        
        init_p_cnt = 1
        init_p_idx = pflist[init_p_cnt]
        # print("top P index: ", init_p_idx)

        she = shelve.open(TRAIN_DIR+"residual.bat") # for Vid4 dataset
        overall_info = she[classname] # residual
        she.close()

        overall_cluster_label = []
        overall_clusered_res = []
        for block_info in overall_info:

            ## obtain 8*8 MV and residual
            block_MV = block_info[0].reshape(-1)
            block_res = block_info[1].reshape(-1)
            cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = block_MV
            if cur_idx > init_p_idx:
                # print("New B frame idx: ", cur_idx)
                # update p index
                init_p_cnt += 1
                if init_p_cnt < len(pflist):
                    init_p_idx = pflist[init_p_cnt]
                    # print("top P index: ", init_p_idx)

                    res_arr = np.array(res_list)

                    ## K-means clustering algorithm
                    if int(res_arr.shape[0]/ratio) != 0:
                        print("Cluster B frames before: ", cur_idx)
                        # ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio),verbose=1,algorithm='full',n_init = 2)
                        ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio),verbose = 1)
                        cluster_label = ms.fit_predict(res_arr)

                        ## calculate the clustered residual
                        clustered_res_list = []
                        clustered_res_list = list(ms.cluster_centers_.astype("float"))

                        overall_cluster_label.append(cluster_label)
                        overall_clusered_res.append(clustered_res_list)
                        MV_list = []
                        res_list = []
                        print("success")

            MV_list.append(block_MV)
            res_list.append(block_res)

        if len(res_list) != 0:
            res_arr = np.array(res_list)

            ## K-means clustering algorithm
            # ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio),verbose=1,algorithm='full',n_init = 2)
            ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio),verbose = 1)
            cluster_label = ms.fit_predict(res_arr)
            
            ## calculate the clustered residual
            clustered_res_list = []
            clustered_res_list = list(ms.cluster_centers_.astype("float"))

            overall_cluster_label.append(cluster_label)
            overall_clusered_res.append(clustered_res_list)
            MV_list = []
            res_list = []
            print("success")

        ## K-means algorithm
        cluster = shelve.open(TRAIN_DIR+"cluster_label_r"+str(ratio)+".bat")
        cluster[classname] = overall_cluster_label
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+"cluster_res_r"+str(ratio)+".bat")
        cluster_res[classname] = overall_clusered_res
        cluster_res.close()
    return

### code for clustering clustered residual
def Cluster_clustered_res_func(ratio_clustered,ratio):
    # print(ratio_clustered,ratio,interval)
    for i in classname_list:
        classname = i
        print("classname: ",classname)
        cluster = shelve.open(TRAIN_DIR+"cluster_label_r"+str(ratio)+".bat")
        overall_cluster_label = cluster[classname]
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"cluster_res_r"+str(ratio)+".bat")
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
            # ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio_clustered),verbose=1,algorithm='full',n_init = 2)
            ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio_clustered),verbose = 1)
            cluster_clustered_label = ms.fit_predict(res_arr) # calculate the clustered residual

            clustered_res_list = []
            clustered_res_list = list(ms.cluster_centers_.astype("float"))

            overall_cluster_clustered_label.append(cluster_clustered_label)
            overall_cluster_clusered_res.append(cluster_clustered_res_list)
        
        #store back
        cluster_tmp = shelve.open(TRAIN_DIR+"cluster_clustered_label_r"+str(ratio)+".bat")
        cluster_tmp[classname] = overall_cluster_clustered_label
        cluster_tmp.close()

        cluster_res_tmp = shelve.open(TRAIN_DIR+"cluster_clustered_res_r"+str(ratio)+".bat")
        cluster_res_tmp[classname] = overall_cluster_clusered_res
        cluster_res_tmp.close()
    return

### code for clustering delta residual = (residual-clustered residual)
def Cluster_delta_func(ratio_delta,ratio):
    for i in classname_list:
        classname = i
        print("classname: ",classname)
        cluster = shelve.open(TRAIN_DIR+"cluster_label_r"+str(ratio)+".bat")
        overall_cluster_label = cluster[classname] # clustered label
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"cluster_res_r"+str(ratio)+".bat")
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
                # ms = KMeans(n_clusters=int(delta_res_arr.shape[0]/ratio_delta),verbose=1,algorithm='full',n_init = 2)
                ms = KMeans(n_clusters=int(delta_res_arr.shape[0]/ratio_delta),verbose = 1)
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

        cluster = shelve.open(TRAIN_DIR+"cluster_delta_label_r"+str(ratio)+".bat")
        cluster[classname] = overall_clustered_delta_label_list
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+"cluster_delta_res_r"+str(ratio)+".bat")
        cluster_res[classname] = overall_clustered_delta_res_list
        cluster_res.close()
        print("success!")
    return

# if __name__ == '__main__':
ratio = int(sys.argv[1])
Cluster_res_func(ratio)
compression_ratio = 16 # compress weights 16x

# ratio_clustered = int(compression_ratio/ratio)
# Cluster_clustered_res_func(ratio_clustered,ratio)

ratio_delta = int(compression_ratio*ratio/(ratio-compression_ratio))
Cluster_delta_func(ratio_delta,ratio)


