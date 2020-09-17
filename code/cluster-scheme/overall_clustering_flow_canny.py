## 针对Vid4数据集；直接对GT-SR的残差进行聚类，聚类时的聚类比例考虑了边缘点的影响
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
TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/Vid4_Cluster/canny/"
TRAIN_res_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/Vid4_Cluster/"
IDX_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/idx/" # idx directory
B_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/bframe_sr_reconstruction/" # SR result
PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/BIx4/" # GT_LR_pic
GT_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/GT/"
MVS_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/mvs/"
residual_info = [] # a list to store all info, including MV and frequency
classname_list = ['calendar','city','foliage','walk']
# classname_list = ['city'] # need to modify!!!!
# MV_list = []
# res_list = []
# MV_list_b = []
# res_list_b = []
# over_res_list = []
# over_res_list_b = []

# reorder the sequence of decoding frame
def ReorderDecFrame(para_mvGroup):

	orderDecFrame = []  #house the index after reoder
	idx = 0  			#index for tranversing orderDecFrame

	for row in para_mvGroup:
		
		referenceIdx = row[1]
		newFrameIdx  = row[0] #the new referenced frame
		
		# Identify whether the reference frame firstly coming out
		if int(referenceIdx) not in orderDecFrame:
			orderDecFrame.insert(idx, int(referenceIdx))
			idx = idx + 1
			
		# Identify  whether the referenced frame firstly coming out
		if int(newFrameIdx) not in orderDecFrame:
			orderDecFrame.insert(idx, int(newFrameIdx))
			idx = idx + 1
		
	return orderDecFrame

# reconstruct B frames using the centroid residuals and the centroid delta residual
def Reconstruct_centroid_res_delta_func(h_ratio,l_ratio,h_ratio_delta,l_ratio_delta):
    for i in classname_list:
        classname = i
        print("classname: ",classname)

        cluster = shelve.open(TRAIN_DIR+"h_cluster_label_r"+str(h_ratio)+".bat")
        h_overall_cluster_label = cluster[classname] # clustered label
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"h_cluster_res_r"+str(h_ratio)+".bat")
        h_overall_centroid_res = cluster_res[classname] # central residual
        cluster_res.close()
        cluster = shelve.open(TRAIN_DIR+"l_cluster_label_r"+str(l_ratio)+".bat")
        l_overall_cluster_label = cluster[classname] # clustered label
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"l_cluster_res_r"+str(l_ratio)+".bat")
        l_overall_centroid_res = cluster_res[classname] # central residual
        cluster_res.close()
        cluster = shelve.open(TRAIN_DIR+"h_cluster_delta_label_r"+str(h_ratio)+".bat")
        h_overall_delta_label_list = cluster[classname]
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"h_cluster_delta_res_r"+str(h_ratio)+".bat")
        h_overall_centroid_delta_res_list = cluster_res[classname]
        cluster_res.close()
        cluster = shelve.open(TRAIN_DIR+"l_cluster_delta_label_r"+str(l_ratio)+".bat")
        l_overall_delta_label_list = cluster[classname]
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"l_cluster_delta_res_r"+str(l_ratio)+".bat")
        l_overall_centroid_delta_res_list = cluster_res[classname]
        cluster_res.close()

        MV_list_cluster = shelve.open(TRAIN_DIR+"motion_vector_list.bat")
        MV_list = MV_list_cluster[classname]
        MV_list_cluster.close()
        MV_list_b_cluster_res = shelve.open(TRAIN_DIR+"motion_vector_list_b.bat")
        MV_list_b = MV_list_b_cluster_res[classname]
        MV_list_b_cluster_res.close()
        over_res_list_cluster = shelve.open(TRAIN_DIR+"res_list.bat")
        over_res_list = over_res_list_cluster[classname]
        over_res_list_cluster.close()
        over_res_list_b_cluster_res = shelve.open(TRAIN_DIR+"res_list_b.bat")
        over_res_list_b = over_res_list_b_cluster_res[classname]
        over_res_list_b_cluster_res.close()
        

        #init frame info
        video_path = PICS_DIR + classname
        pic_names = os.listdir(video_path)
        frame_num = len(pic_names)
        img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_h, frame_w, _ = img.shape
        sr_frame_h = 4 * frame_h #576
        sr_frame_w = 4 * frame_w #720
        frame_mat_SR = np.zeros((frame_num,sr_frame_h,sr_frame_w, 3), dtype="uint8") #init frame_mat
        for hh in range(frame_num):
            # print(B_DIR + classname + "/%08d.png" % i)
            tmp_img = cv2.imread(B_DIR + classname + "/%08d.png" % hh) # read SR result
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            frame_mat_SR[hh] = tmp_img

        #generate depending order
        mvName_file = MVS_DIR+classname+".csv"
        csvMvFile = open(mvName_file)
        csvMvReader = csv.reader(csvMvFile)
        mvGroup = list(csvMvReader)	
        resultReoderIndex = ReorderDecFrame(mvGroup) # a list for storing decoding order
        # print(resultReoderIndex)
        bflist = []

        with open(IDX_DIR+"b/"+classname, "r") as file:
            for row in file:
                bflist.append(int(row)-1)


        for idx_order in resultReoderIndex:
            if idx_order in bflist:
                B_img = np.stack([np.zeros((sr_frame_h,sr_frame_w),'int'), np.zeros((sr_frame_h,sr_frame_w),'int'), np.zeros((sr_frame_h,sr_frame_w),'int')*255], axis=2).astype("float")
                # B_img = cv2.imread(B_DIR + classname + '/' + "%08d.png" % idx_order, -1)
                # B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB).astype("float")

                # print("reconstruct a B frame: ",idx_order)

                begin_cnt = 0 # use to locate MV in block_MV_list
                for label_cnt in range(len(h_overall_cluster_label)):
                    cluster_label = h_overall_cluster_label[label_cnt]
                    centroid_res = h_overall_centroid_res[label_cnt]
                    cluster_delta_label = h_overall_delta_label_list[label_cnt] # delta label
                    centroid_delta_res = h_overall_centroid_delta_res_list[label_cnt] # delta residual
                    for label_idx, label_data in enumerate(cluster_label):
                        # locate MV
                        MV_idx = label_idx + begin_cnt
                        tmp_MV = MV_list[MV_idx]
                        cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = tmp_MV
                        if cur_idx == idx_order:
                            tmp_cur_idx = cur_idx
                            single_res = centroid_res[label_data] #shape = (1,192)

                            # locate delta res
                            delta_label_data = cluster_delta_label[label_idx]
                            if h_ratio_delta != 0:
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
                                        if (0<=(refx + px) < sr_frame_w) and (0<=(refy+py) < sr_frame_h):
                                            B_img[cury+py,curx+px,:] = ref_frame_sr[refy+py,refx + px,:].astype("float") + tmp_res[py,px,:].astype("float")
                                        else:
                                            B_img[cury+py,curx+px,:] = tmp_res[py,px,:].astype("float")
                                            # B_img[cury+py,curx+px,:] = np.array([0,0,255]) # need to modify
                                            # B_img[cury+py,curx+px,pc] = 255
                        
                    begin_cnt += len(h_overall_cluster_label[label_cnt])

                begin_cnt = 0 # use to locate MV in block_MV_list
                for label_cnt in range(len(l_overall_cluster_label)):
                    cluster_label = l_overall_cluster_label[label_cnt]
                    centroid_res = l_overall_centroid_res[label_cnt]
                    cluster_delta_label = l_overall_delta_label_list[label_cnt] # delta label
                    centroid_delta_res = l_overall_centroid_delta_res_list[label_cnt] # delta residual
                    for label_idx, label_data in enumerate(cluster_label):
                        # locate MV
                        MV_idx = label_idx + begin_cnt
                        tmp_MV = MV_list_b[MV_idx]
                        cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = tmp_MV
                        if cur_idx == idx_order:
                            single_res = centroid_res[label_data] #shape = (1,192)

                            # locate delta res
                            delta_label_data = cluster_delta_label[label_idx]
                            if l_ratio_delta != 0:
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
                        
                    begin_cnt += len(l_overall_cluster_label[label_cnt])

                for i in range(sr_frame_w):
                    for j in range(sr_frame_h):
                        for c in range(3):
                            if B_img[j,i,c]<0:
                                B_img[j,i,c] = 0
                            if B_img[j,i,c]>255:
                                B_img[j,i,c] = 255
                                
                B_img = B_img.astype("uint8")
                frame_mat_SR[idx_order] = B_img # notice!!!
                B_img = cv2.cvtColor(B_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(B_DIR + classname + '/' + "%08d.png" % idx_order, B_img)
                # B_img = np.stack([np.zeros((sr_frame_h,sr_frame_w),'int'), np.zeros((sr_frame_h,sr_frame_w),'int'), np.zeros((sr_frame_h,sr_frame_w),'int')*255], axis=2).astype("float")

### code for clustering 8*8 residual blocks
def Cluster_res_func(h_ratio,l_ratio):

    

    for i in classname_list:
        MV_list = []
        res_list = []
        MV_list_b = []
        res_list_b = []
        over_res_list = []
        over_res_list_b = []

        classname = i

        video_path = PICS_DIR + classname
        pic_names = os.listdir(video_path)
        frame_num = len(pic_names)
        tmp_img_hh = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
        img = cv2.cvtColor(tmp_img_hh, cv2.COLOR_BGR2RGB)
        frame_h, frame_w, _ = img.shape
        sr_frame_h = 4 * frame_h
        sr_frame_w = 4 * frame_w
        canny_output_img = np.zeros((frame_num,sr_frame_h,sr_frame_w), dtype="uint8") #init frame_mat
        for frame_cnt in range(frame_num):
            canny_input_img = cv2.imread(GT_DIR+classname+"/%08d.png" % frame_cnt)
            canny_output_img[frame_cnt] = cv2.Canny(canny_input_img,150,250) # image after canny
            # print(canny_output_img[frame_cnt].shape)

        total_block = 0
        total_boundary = 0
        total_non_boundary = 0
        
        print("classname: ",classname)
        pflist = [] # a list to store the index of P frames

        ## load idx
        with open(IDX_DIR+"p/"+classname, "r") as file:
            for row in file:
                pflist.append(int(row)-1)
        
        init_p_cnt = 1
        init_p_idx = pflist[init_p_cnt]


        she = shelve.open(TRAIN_res_DIR+"residual.bat") # for Vid4 dataset
        residual_info = she[classname] # residual
        she.close()

        overall_cluster_label = [] # label information
        overall_centroid_res = [] # centroid information
        overall_cluster_label_b = [] # label information
        overall_centroid_res_b = [] # centroid information
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

                    res_arr = np.array(res_list) #non-boundary array, need high compression
                    res_arr_b = np.array(res_list_b) # boundary array

                    ## K-means clustering algorithm
                    if int(res_arr.shape[0]/h_ratio) != 0: # non-boudary clustering
                        # print("Cluster B frames before: ", cur_idx)
                        # ms = KMeans(n_clusters=int(res_arr.shape[0]/h_ratio),verbose=1,algorithm='full',n_init = 2)
                        ms = KMeans(n_clusters=int(res_arr.shape[0]/h_ratio),n_init = 2)
                        cluster_label = ms.fit_predict(res_arr) # the label of residual inside an interval

                        ## calculate the clustered residual
                        centroid_res_list = []
                        centroid_res_list = list(ms.cluster_centers_.astype("float"))

                        overall_cluster_label.append(cluster_label)
                        overall_centroid_res.append(centroid_res_list)
                        res_list = []

                    ## K-means clustering algorithm
                    if int(res_arr_b.shape[0]/l_ratio) != 0: # boudary clustering
                        # print("Cluster B frames before: ", cur_idx)
                        # ms = KMeans(n_clusters=int(res_arr.shape[0]/l_ratio),verbose=1,algorithm='full',n_init = 2)
                        ms = KMeans(n_clusters=int(res_arr_b.shape[0]/l_ratio),n_init = 2)
                        cluster_label = ms.fit_predict(res_arr_b) # the label of residual inside an interval

                        ## calculate the clustered residual
                        centroid_res_list = []
                        centroid_res_list = list(ms.cluster_centers_.astype("float"))

                        overall_cluster_label_b.append(cluster_label)
                        overall_centroid_res_b.append(centroid_res_list)
                        res_list_b = []


            cnt_non_zero = 0
            cnt_non_zero = np.sum(canny_output_img[cur_idx,cury:cury+8, curx:curx+8]!=0)
            ratio_non_zero = float(cnt_non_zero)/float(64)
            if ratio_non_zero >= 0.4: # boundary
                MV_list_b.append(block_MV)
                res_list_b.append(block_res)
                over_res_list_b.append(block_res)
                total_boundary += 1
            else: # non-boundary
                MV_list.append(block_MV)
                res_list.append(block_res)
                over_res_list.append(block_res)
                total_non_boundary += 1
            total_block += 1


        if len(res_list) != 0:
            # process non-boudary block
            res_arr = np.array(res_list)
            ## K-means clustering algorithm
            if int(res_arr.shape[0]/h_ratio) != 0:
                ms = KMeans(n_clusters=int(res_arr.shape[0]/h_ratio),n_init = 2)
                cluster_label = ms.fit_predict(res_arr)
                ## calculate the clustered residual
                centroid_res_list = []
                centroid_res_list = list(ms.cluster_centers_.astype("float"))
                overall_cluster_label.append(cluster_label)
                overall_centroid_res.append(centroid_res_list)
                res_list = []

        if len(res_list_b) != 0:
            # process boudary block
            res_arr_b = np.array(res_list_b)
            ## K-means clustering algorithm
            if int(res_arr_b.shape[0]/l_ratio) != 0:
                ms = KMeans(n_clusters=int(res_arr_b.shape[0]/l_ratio),n_init = 2)
                cluster_label = ms.fit_predict(res_arr_b)
                ## calculate the clustered residual
                centroid_res_list = []
                centroid_res_list = list(ms.cluster_centers_.astype("float"))
                overall_cluster_label_b.append(cluster_label)
                overall_centroid_res_b.append(centroid_res_list)
                res_list_b = []

        print("ratio = ")
        print(float(total_boundary)/float(total_block))

        ## K-means algorithm
        cluster = shelve.open(TRAIN_DIR+"h_cluster_label_r"+str(h_ratio)+".bat")
        cluster[classname] = overall_cluster_label
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+"h_cluster_res_r"+str(h_ratio)+".bat")
        cluster_res[classname] = overall_centroid_res
        cluster_res.close()

        ## K-means algorithm
        cluster = shelve.open(TRAIN_DIR+"l_cluster_label_r"+str(l_ratio)+".bat")
        cluster[classname] = overall_cluster_label_b
        cluster.close()

        cluster_res = shelve.open(TRAIN_DIR+"l_cluster_res_r"+str(l_ratio)+".bat")
        cluster_res[classname] = overall_centroid_res_b
        cluster_res.close()

        ## K-means algorithm
        MV_list_cluster = shelve.open(TRAIN_DIR+"motion_vector_list.bat")
        MV_list_cluster[classname] = MV_list
        MV_list_cluster.close()
        MV_list_b_cluster_res = shelve.open(TRAIN_DIR+"motion_vector_list_b.bat")
        MV_list_b_cluster_res[classname] = MV_list_b
        MV_list_b_cluster_res.close()
        over_res_list_cluster = shelve.open(TRAIN_DIR+"res_list.bat")
        over_res_list_cluster[classname] = over_res_list
        over_res_list_cluster.close()
        over_res_list_b_cluster_res = shelve.open(TRAIN_DIR+"res_list_b.bat")
        over_res_list_b_cluster_res[classname] = over_res_list_b
        over_res_list_b_cluster_res.close()
    return

### code for clustering delta residual = (residual-clustered residual)
def Cluster_delta_func(h_ratio,l_ratio,h_ratio_delta,l_ratio_delta):
    for i in classname_list:
        classname = i
        print("classname: ",classname)
        cluster = shelve.open(TRAIN_DIR+"h_cluster_label_r"+str(h_ratio)+".bat")
        h_overall_cluster_label = cluster[classname] # clustered label
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"h_cluster_res_r"+str(h_ratio)+".bat")
        h_overall_centroid_res = cluster_res[classname] # central residual
        cluster_res.close()
        cluster = shelve.open(TRAIN_DIR+"l_cluster_label_r"+str(l_ratio)+".bat")
        l_overall_cluster_label = cluster[classname] # clustered label
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"l_cluster_res_r"+str(l_ratio)+".bat")
        l_overall_centroid_res = cluster_res[classname] # central residual
        cluster_res.close()

        MV_list_cluster = shelve.open(TRAIN_DIR+"motion_vector_list.bat")
        MV_list = MV_list_cluster[classname]
        MV_list_cluster.close()
        MV_list_b_cluster_res = shelve.open(TRAIN_DIR+"motion_vector_list_b.bat")
        MV_list_b = MV_list_b_cluster_res[classname]
        MV_list_b_cluster_res.close()
        over_res_list_cluster = shelve.open(TRAIN_DIR+"res_list.bat")
        over_res_list = over_res_list_cluster[classname]
        over_res_list_cluster.close()
        over_res_list_b_cluster_res = shelve.open(TRAIN_DIR+"res_list_b.bat")
        over_res_list_b = over_res_list_b_cluster_res[classname]
        over_res_list_b_cluster_res.close()


        begin_cnt = 0 # use to locate MV in block_MV_list and block_res_list
        h_overall_delta_res_list = [] # use to store delta residual

        # calculate the delta of central residual and residual
        for label_cnt in range(len(h_overall_cluster_label)):
            delta_res_list = []
            cluster_label = h_overall_cluster_label[label_cnt]
            centroid_res = h_overall_centroid_res[label_cnt]
            for label_idx, label_data in enumerate(cluster_label): # 0~16000 item number
                # locate MV
                MV_idx = label_idx + begin_cnt
                tmp_MV = MV_list[MV_idx]
                tmp_res = over_res_list[MV_idx]

                # locate res
                single_res = centroid_res[label_data] #shape = (1,192)
                delta_res = tmp_res - single_res
                delta_res_list.append(delta_res)
            h_overall_delta_res_list.append(delta_res_list) # a list storing the delta of central residual and residual
            begin_cnt += len(h_overall_cluster_label[label_cnt])

        
        # clustering the delta of central residual and residual, then calculate the central delta residual
        h_overall_delta_label_list = []
        h_overall_centroid_delta_res_list = []
        for label_cnt in range(len(h_overall_delta_res_list)):
            delta_res_arr = np.array(h_overall_delta_res_list[label_cnt])
            # print(delta_res_arr.shape)
            if h_ratio_delta !=0:
                ms = KMeans(n_clusters=int(delta_res_arr.shape[0]/h_ratio_delta),n_init = 2)
                delta_cluster_label = ms.fit_predict(delta_res_arr)

                clustered_delta_res_list = []
                clustered_delta_res_list = list(ms.cluster_centers_.astype("float"))                     

                h_overall_delta_label_list.append(delta_cluster_label)
                h_overall_centroid_delta_res_list.append(clustered_delta_res_list)
            
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

                h_overall_delta_label_list.append(delta_cluster_label)
                h_overall_centroid_delta_res_list.append(clustered_delta_res_list)
        
        
        begin_cnt = 0 # use to locate MV in block_MV_list and block_res_list
        l_overall_delta_res_list = [] # use to store delta residual

        # calculate the delta of central residual and residual
        for label_cnt in range(len(l_overall_cluster_label)):
            delta_res_list = []
            cluster_label = l_overall_cluster_label[label_cnt]
            centroid_res = l_overall_centroid_res[label_cnt]
            for label_idx, label_data in enumerate(cluster_label): # 0~16000 item number
                # locate MV
                MV_idx = label_idx + begin_cnt
                tmp_MV = MV_list_b[MV_idx]
                tmp_res = over_res_list_b[MV_idx]

                # locate res
                single_res = centroid_res[label_data] #shape = (1,192)
                delta_res = tmp_res - single_res
                delta_res_list.append(delta_res)
            l_overall_delta_res_list.append(delta_res_list) # a list storing the delta of central residual and residual
            begin_cnt += len(l_overall_cluster_label[label_cnt])

        
        # clustering the delta of central residual and residual, then calculate the central delta residual
        l_overall_delta_label_list = []
        l_overall_centroid_delta_res_list = []
        for label_cnt in range(len(l_overall_delta_res_list)):
            delta_res_arr = np.array(l_overall_delta_res_list[label_cnt])
            # print(delta_res_arr.shape)
            if l_ratio_delta !=0:
                ms = KMeans(n_clusters=int(delta_res_arr.shape[0]/l_ratio_delta),n_init = 2)
                delta_cluster_label = ms.fit_predict(delta_res_arr)

                clustered_delta_res_list = []
                clustered_delta_res_list = list(ms.cluster_centers_.astype("float"))                     

                l_overall_delta_label_list.append(delta_cluster_label)
                l_overall_centroid_delta_res_list.append(clustered_delta_res_list)
            
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

                l_overall_delta_label_list.append(delta_cluster_label)
                l_overall_centroid_delta_res_list.append(clustered_delta_res_list)




        cluster = shelve.open(TRAIN_DIR+"h_cluster_delta_label_r"+str(h_ratio)+".bat")
        cluster[classname] = h_overall_delta_label_list
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"h_cluster_delta_res_r"+str(h_ratio)+".bat")
        cluster_res[classname] = h_overall_centroid_delta_res_list
        cluster_res.close()
        cluster = shelve.open(TRAIN_DIR+"l_cluster_delta_label_r"+str(l_ratio)+".bat")
        cluster[classname] = l_overall_delta_label_list
        cluster.close()
        cluster_res = shelve.open(TRAIN_DIR+"l_cluster_delta_res_r"+str(l_ratio)+".bat")
        cluster_res[classname] = l_overall_centroid_delta_res_list
        cluster_res.close()
        # print("success!")

    
    return


#begin func
h_ratio = int(sys.argv[1])
l_ratio = int(sys.argv[2])
h_compression_ratio = 32 # compress weights 32x
l_compression_ratio = 16 # compress weights 16x
h_ratio_delta = int(h_compression_ratio*h_ratio/(h_ratio-h_compression_ratio))
l_ratio_delta = int(l_compression_ratio*l_ratio/(l_ratio-l_compression_ratio))
print(h_ratio_delta,l_ratio_delta)
### first floor clustering
Cluster_res_func(h_ratio,l_ratio)
### second floor clustering
Cluster_delta_func(h_ratio,l_ratio,h_ratio_delta,l_ratio_delta)


### reconstruction
residual_info = [] # init
Reconstruct_centroid_res_delta_func(h_ratio,l_ratio,h_ratio_delta,l_ratio_delta)


