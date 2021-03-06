## 针对Vimeo90K数据集；直接对GT-SR的残差进行聚类
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


#for Vimeo90K dataset
IDX_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/" #+foldername+idx
MVS_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/" #+foldername+mvs
PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/BIx4/" # GT_LR_pic, +foldername
HR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/GT/" # GT_HR_pic, +foldername
SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/SR_result/" # +foldername
B_DIR = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/Our_result/" # +foldername
TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/Vimeo90K_Cluster/" # +foldername

IDX_DIR1="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/" #+foldername+idx
MVS_DIR1="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/" #+foldername+mvs
PICS_DIR1 = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/BIx4/" # GT_LR_pic, +foldername
HR_PICS_DIR1 = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/GT/" # GT_HR_pic, +foldername
SR_PICS_DIR1 = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/SR_result/" # +foldername
B_DIR1 = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/Our_result/" # +foldername
TRAIN_DIR1 = "/home/songzhuoran/video/video-sr-acc/train_info/Vimeo90K_Cluster/" # +foldername
# folder_list = ['00027', '00029', '00053', '00077', '00093', '00049', '00044', '00015', '00056', '00067', '00024', '00072', '00023', '00060', '00011', '00087', '00012', '00026', '00064', '00047', '00004', '00083', '00019', '00074', '00076', '00057', '00034', '00091', '00065', '00014', '00050', '00018', '00089', '00022', '00009', '00068', '00082', '00088', '00025', '00003', '00090', '00046', '00069', '00063', '00028', '00058', '00005', '00001', '00002', '00040', '00035', '00032', '00094', '00062', '00037', '00092', '00008', '00016', '00095', '00054', '00010', '00078', '00073', '00041', '00039', '00070', '00017', '00066', '00055', '00084', '00006', '00096', '00085', '00048', '00080', '00075', '00030', '00042', '00021', '00071', '00079', '00031', '00051', '00013', '00036', '00045', '00043', '00081', '00052', '00007', '00086', '00033', '00038', '00061']
folder_list = ['00027']
foldername = '00027'
classname_list = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029']
classname = '000'
residual_info = [] # a list to store all info, including MV and frequency
MV_list = []
res_list = []

# reconstruct B frames using the centroid residuals and the centroid delta residual
def Reconstruct_centroid_res_delta_func(ratio_delta,ratio):
    for i in classname_list:
        classname = i
        print("classname: ",classname)

        cluster = shelve.open(TRAIN_DIR+classname+"cluster_label_r"+str(ratio)+".bat")
        overall_cluster_label = cluster[classname]
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+classname+"cluster_res_r"+str(ratio)+".bat")
        overall_centroid_res = cluster_res[classname]
        cluster_res.close()
        she = shelve.open(TRAIN_DIR+"residual_"+classname+".bat")
        residual_info = she[classname]
        she.close()

        cluster = shelve.open(TRAIN_DIR+classname+"cluster_delta_label_r"+str(ratio)+".bat")
        overall_delta_label_list = cluster[classname]
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+classname+"cluster_delta_res_r"+str(ratio)+".bat")
        overall_centroid_delta_res_list = cluster_res[classname]
        cluster_res.close()
        

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
            tmp_img = cv2.imread(B_DIR + classname + "/im%01d.png" % (i+1)) # read SR result
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            frame_mat_SR[i] = tmp_img


        block_MV_list = []
        # print(len(overall_cluster_label))
        for block_info in residual_info:
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
            centroid_res = overall_centroid_res[label_cnt]
            cluster_delta_label = overall_delta_label_list[label_cnt] # delta label
            centroid_delta_res = overall_centroid_delta_res_list[label_cnt] # delta residual
            for label_idx, label_data in enumerate(cluster_label):
                # locate MV
                MV_idx = label_idx + begin_cnt
                tmp_MV = block_MV_list[MV_idx]
                cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = tmp_MV

                #read and write a B frame
                if tmp_cur_idx == -1:
                    print("init a B frame, the idx is %d: ", cur_idx)
                    B_img = cv2.imread(B_DIR + classname + '/' + "/im%01d.png" % (cur_idx+1), -1)
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
                    frame_mat_SR[tmp_cur_idx] = B_img # notice!!!
                    B_img = cv2.cvtColor(B_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(B_DIR + classname + '/' + "/im%01d.png" % (tmp_cur_idx+1), B_img)
                    print("init a B frame, the idx is %d: ", cur_idx)
                    B_img = cv2.imread(B_DIR + classname + '/' + "/im%01d.png" % (cur_idx+1), -1)
                    B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB).astype("float")
                tmp_cur_idx = cur_idx

                # locate res
                single_res = centroid_res[label_data] #shape = (1,192)

                # locate delta res
                delta_label_data = cluster_delta_label[label_idx]
                if ratio_delta != 0:
                    delta_res = centroid_delta_res[delta_label_data]
                else:
                    delta_res = centroid_delta_res[int(delta_label_data)]
                

                tmp_res = single_res + delta_res
                tmp_res = tmp_res.reshape((block_h,block_w,3))


                #reconstruction
                ref_frame_sr = frame_mat_SR[ref_idx]
                for px in range(block_w):
                    for py in range(block_h):
                        if ((curx+px) < sr_frame_w) and ((cury+py) < sr_frame_h):
                            if (0 <= (refx + px) < sr_frame_w) and (0 <= (refy+py) < sr_frame_h):
                                B_img[cury+py,curx+px,:] = ref_frame_sr[refy+py,refx + px,:].astype("float") + tmp_res[py,px,:].astype("float")
                            else:
                                B_img[cury+py,curx+px,:] = tmp_res[py,px,:].astype("float")
                                # B_img[cury+py,curx+px,:] = np.array([0,0,255]) # need to modify
                                # B_img[cury+py,curx+px,pc] = 255

                
            begin_cnt += len(overall_cluster_label[label_cnt])

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

        she = shelve.open(TRAIN_DIR+"residual_"+classname+".bat") # for Vid4 dataset
        residual_info = she[classname] # residual
        she.close()

        overall_cluster_label = [] # label information
        overall_centroid_res = [] # centroid information
        for block_info in residual_info:

            ## obtain 8*8 MV and residual
            block_MV = block_info[0].reshape(-1) # (1,192)
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
                        ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio),n_init = 2)
                        cluster_label = ms.fit_predict(res_arr) # the label of residual inside an interval

                        ## calculate the clustered residual
                        centroid_res_list = []
                        centroid_res_list = list(ms.cluster_centers_.astype("float"))

                        overall_cluster_label.append(cluster_label)
                        overall_centroid_res.append(centroid_res_list)
                        MV_list = []
                        res_list = []
                        print("success")

            MV_list.append(block_MV)
            res_list.append(block_res)

        if len(res_list) != 0:
            res_arr = np.array(res_list)

            ## K-means clustering algorithm
            # ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio),verbose=1,algorithm='full',n_init = 2)
            ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio),n_init = 2)
            cluster_label = ms.fit_predict(res_arr)
            
            ## calculate the clustered residual
            centroid_res_list = []
            centroid_res_list = list(ms.cluster_centers_.astype("float"))

            overall_cluster_label.append(cluster_label)
            overall_centroid_res.append(centroid_res_list)
            MV_list = []
            res_list = []
            print("success")

        ## K-means algorithm
        cluster = shelve.open(TRAIN_DIR+classname+"cluster_label_r"+str(ratio)+".bat")
        cluster[classname] = overall_cluster_label
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+classname+"cluster_res_r"+str(ratio)+".bat")
        cluster_res[classname] = overall_centroid_res
        cluster_res.close()
    return

### code for clustering clustered residual
def Cluster_clustered_res_func(ratio_clustered,ratio):
    # print(ratio_clustered,ratio,interval)
    for i in classname_list:
        classname = i
        print("classname: ",classname)
        cluster = shelve.open(TRAIN_DIR+classname+"cluster_label_r"+str(ratio)+".bat")
        overall_cluster_label = cluster[classname]
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+classname+"cluster_res_r"+str(ratio)+".bat")
        overall_centroid_res = cluster_res[classname]
        cluster_res.close()

        overall_cluster_clustered_label = []
        overall_cluster_clusered_res = []

        for label_cnt in range(len(overall_cluster_label)):
            cluster_label = overall_cluster_label[label_cnt]
            centroid_res = overall_centroid_res[label_cnt]
            # print(centroid_res[0].shape)
            res_list = []
            for label_idx, label_data in enumerate(cluster_label):
                tmp_res = centroid_res[label_data]
                res_list.append(tmp_res)
            #apply K-means to cluster clustered residual
            res_arr = np.array(res_list)
            # ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio_clustered),verbose=1,algorithm='full',n_init = 2)
            ms = KMeans(n_clusters=int(res_arr.shape[0]/ratio_clustered),n_init = 2)
            cluster_clustered_label = ms.fit_predict(res_arr) # calculate the clustered residual

            centroid_res_list = []
            centroid_res_list = list(ms.cluster_centers_.astype("float"))

            overall_cluster_clustered_label.append(cluster_clustered_label)
            overall_cluster_clusered_res.append(cluster_clustered_res_list)
        
        #store back
        cluster_tmp = shelve.open(TRAIN_DIR+classname+"cluster_clustered_label_r"+str(ratio)+".bat")
        cluster_tmp[classname] = overall_cluster_clustered_label
        cluster_tmp.close()

        cluster_res_tmp = shelve.open(TRAIN_DIR+classname+"cluster_clustered_res_r"+str(ratio)+".bat")
        cluster_res_tmp[classname] = overall_cluster_clusered_res
        cluster_res_tmp.close()
    return

### code for clustering delta residual = (residual-clustered residual)
def Cluster_delta_func(ratio_delta,ratio):
    for i in classname_list:
        classname = i
        print("classname: ",classname)
        cluster = shelve.open(TRAIN_DIR+classname+"cluster_label_r"+str(ratio)+".bat")
        overall_cluster_label = cluster[classname] # clustered label
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+classname+"cluster_res_r"+str(ratio)+".bat")
        overall_centroid_res = cluster_res[classname] # central residual
        cluster_res.close()
        she = shelve.open(TRAIN_DIR+"residual_"+classname+".bat") # all residual for 8*8 blocks
        residual_info = she[classname]
        she.close()

        block_MV_list = []
        block_res_list = []
        for block_info in residual_info:
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
            centroid_res = overall_centroid_res[label_cnt]
            for label_idx, label_data in enumerate(cluster_label): # 0~16000 item number
                # locate MV
                MV_idx = label_idx + begin_cnt
                tmp_MV = block_MV_list[MV_idx]
                tmp_res = block_res_list[MV_idx]

                # locate res
                single_res = centroid_res[label_data] #shape = (1,192)
                delta_res = tmp_res - single_res
                delta_res_list.append(delta_res)
            overall_delta_res_list.append(delta_res_list) # a list storing the delta of central residual and residual
            begin_cnt += len(overall_cluster_label[label_cnt])

        
        # clustering the delta of central residual and residual, then calculate the central delta residual
        overall_delta_label_list = []
        overall_centroid_delta_res_list = []
        for label_cnt in range(len(overall_delta_res_list)):
            delta_res_arr = np.array(overall_delta_res_list[label_cnt])
            print(delta_res_arr.shape)
            if ratio_delta !=0:
                # ms = KMeans(n_clusters=int(delta_res_arr.shape[0]/ratio_delta),verbose=1,algorithm='full',n_init = 2)
                ms = KMeans(n_clusters=int(delta_res_arr.shape[0]/ratio_delta),n_init = 2)
                delta_cluster_label = ms.fit_predict(delta_res_arr)

                clustered_delta_res_list = []
                clustered_delta_res_list = list(ms.cluster_centers_.astype("float"))                     

                overall_delta_label_list.append(delta_cluster_label)
                overall_centroid_delta_res_list.append(clustered_delta_res_list)
            
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

                overall_delta_label_list.append(delta_cluster_label)
                overall_centroid_delta_res_list.append(clustered_delta_res_list)

        cluster = shelve.open(TRAIN_DIR+classname+"cluster_delta_label_r"+str(ratio)+".bat")
        cluster[classname] = overall_delta_label_list
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+classname+"cluster_delta_res_r"+str(ratio)+".bat")
        cluster_res[classname] = overall_centroid_delta_res_list
        cluster_res.close()
        print("success!")
    return


#begin func
ratio = int(sys.argv[1])
compression_ratio = 16 # compress weights 16x
ratio_delta = int(compression_ratio*ratio/(ratio-compression_ratio))
print(ratio_delta)

for foldername in folder_list:
    print("foldername: ",foldername)
    IDX_DIR=IDX_DIR1+foldername+'/idx/' #+foldername+idx
    MVS_DIR=MVS_DIR1+foldername+'/mvs/' #+foldername+mvs
    PICS_DIR=PICS_DIR1+foldername+'/' # GT_LR_pic, +foldername
    HR_PICS_DIR = HR_PICS_DIR1+foldername+'/' # GT_HR_pic, +foldername
    SR_PICS_DIR = SR_PICS_DIR1+foldername+'/' # +foldername
    TRAIN_DIR = TRAIN_DIR1+foldername+'/' # +foldername
    B_DIR = B_DIR1+foldername+'/' # +foldername

    classname_list = os.listdir(PICS_DIR)
    # print(classname_list)
    ### first floor clustering
    Cluster_res_func(ratio)
    ### second floor clustering
    Cluster_delta_func(ratio_delta,ratio)


    ### reconstruction
    residual_info = [] # init
    Reconstruct_centroid_res_delta_func(ratio_delta,ratio)


