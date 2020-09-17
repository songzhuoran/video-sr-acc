### code for generating 8*8 residual blocks, waiting for clustering
## I/P帧为ground truth，用于与传统encoder比较
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


#整帧地进行SR

# #for Vid4 dataset
# IDX_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/idx/"
# MVS_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/mvs/"
# B_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/bframe_sr/" # SR result
# RES_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/Residuals/"
# MV_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/mvs/"
# ORDER_DIR="/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/order/"
# PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/BIx4/" # GT_LR_pic
# HR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/GT/" # GT_HR_pic
# SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/SR_result/"
# TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/Vid4_Cluster/"
# classname_list = ['calendar','city','foliage','walk']
# classname = 'calendar'

# #for REDS dataset
# IDX_DIR="/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4/idx/"
# MVS_DIR="/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4/mvs/"
# B_DIR="/home/songzhuoran/video/video-sr-acc/REDS/Our_result/bframe_sr/" # SR result
# RES_DIR="/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4/Residuals/"
# MV_DIR="/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4/mvs/"
# ORDER_DIR="/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4/order/"
# PICS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/BIx4/" # GT_LR_pic
# HR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/GT/" # GT_HR_pic
# SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/SR_result/"
# TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/REDS_Cluster/"
# classname_list = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029']
# classname = '000'

#for Vimeo90k dataset
IDX_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/idx/"
MVS_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/mvs/"
B_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Our_result/bframe_sr/" # SR result
RES_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/Residuals/"
MV_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/mvs/"
ORDER_DIR="/home/songzhuoran/video/video-sr-acc/Vimeo90K/Info_BIx4/order/"
PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/BIx4/" # GT_LR_pic
HR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/GT/" # GT_HR_pic
SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vimeo90K/SR_result/"
TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/Vimeo90K_Cluster/"
classname_list = ['00001','00002','00003','00004','00005','00006','00007','00008','00009','00010','00011','00012','00013','00014','00015','00016','00017','00018','00019','00020','00021','00022','00023','00024','00025','00026','00027','00028','00029','00030','00031','00032','00033','00034','00035','00036','00037','00038','00039','00040','00041','00042','00043','00044','00045','00046','00047','00048','00049','00050','00051','00052','00053','00054','00055','00056','00057','00058','00059','00060','00061','00062','00063','00064','00065','00066','00067','00068','00069','00070','00071','00072','00073','00074','00075','00076','00077','00078','00079','00080','00081','00082','00083','00084','00085','00086','00087','00088','00089','00090','00091','00092','00093','00094','00095','00096']
classname = '00001'

mvsmat = {} # 记录各帧depending关系的dict
bflist = []  # aka b frame list
pflist = []  # aka b frame list
Res_data = [] # a list to store Residual_data
MV_data = [] # a list to store MV
overall_info = [] # a list to store all info, including MV and frequency
video_path = PICS_DIR + classname
pic_names = os.listdir(video_path)
frame_num = len(pic_names)
vis = [False] * frame_num
#get shape and length of the whole video
img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
frame_h, frame_w, _ = img.shape
sr_frame_h = 4 * frame_h
sr_frame_w = 4 * frame_w
frame_mat_GT_HR = np.zeros((frame_num,sr_frame_h,sr_frame_w, 3), dtype="uint8")#init frame_mat
frame_mat_GT_LR = np.zeros((frame_num,frame_h,frame_w, 3), dtype="uint8")
frame_mat_SR_HR = np.zeros((frame_num,sr_frame_h,sr_frame_w, 3), dtype="uint8")


def fetch_MV_data():
    #load the whole res_data file into a py list. prevent redundant loading
    with open(MV_DIR +classname + ".csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            MV_data.append(item)
    # print('res_len', len(Res_data))

def dep_tree_gen():
    # 生成depending tree(一个记录着帧之间的关联性的dict)
    with open(IDX_DIR+"b/"+classname, "r") as file:
        for row in file:
            bflist.append(int(row)-1)
        # print(bflist)

    with open(IDX_DIR+"p/"+classname, "r") as file:
        for row in file:
            pflist.append(int(row)-1)
    for i in range(frame_num):
        mvsmat[i] = set()  #记录每一帧的还原需要哪些帧的数据
        frame_mat_GT_HR[i] = cv2.imread(HR_PICS_DIR + classname + "/%08d.png" % i) # read GT_HR result, bgr
        frame_mat_GT_HR[i]=cv2.cvtColor(frame_mat_GT_HR[i], cv2.COLOR_BGR2RGB)
        frame_mat_GT_LR[i] = cv2.imread(PICS_DIR + classname + "/%08d.png" % i) # read GT_LR result, bgr formate
        frame_mat_GT_LR[i]=cv2.cvtColor(frame_mat_GT_LR[i], cv2.COLOR_BGR2RGB) # convert to rgb
        frame_mat_SR_HR[i] = cv2.imread(SR_PICS_DIR + classname + "/%08d.png" % i) # read GT_LR result, bgr formate
        frame_mat_SR_HR[i]=cv2.cvtColor(frame_mat_SR_HR[i], cv2.COLOR_BGR2RGB) # convert to rgb
    with open(MVS_DIR+classname+".csv","r") as file:
        datainfo = csv.reader(file)
        for row in datainfo:
            mvsmat[int(row[0])].add(int(row[1]))
    return


def bframe_gen_kernel(fcnt):
    global MV_data  #函数内使用全局变量global
    global overall_info
    for row in MV_data:
        if int(float(row[0])) == fcnt:
            cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = np.array(row[0:8]).astype(float).astype(int)
            #generate residual of ground-truth frames
            gt_residual = np.zeros((4*block_h, 4*block_w, 3))
            ref_frame_sr = frame_mat_SR_HR[ref_idx]
            b_frame_gt = frame_mat_GT_HR[cur_idx]
            for px in range(block_w):
                for py in range(block_h):
                    if (curx+px in range(frame_w)) and (cury+py in range(0, frame_h)):
                        # cur块在范围内
                        sr_curpx = 4 * (curx + px) # sr后图片内各点对应的坐标
                        sr_curpy = 4 * (cury + py)
                        sr_refpx = 4 * (refx + px)
                        sr_refpy = 4 * (refy + py)
                        b_block_gt = b_frame_gt[sr_curpy:sr_curpy+4, sr_curpx:sr_curpx+4]
                        if (refx + px in range(0, frame_w)) and (refy+py in range(0, frame_h)):
                            ref = ref_frame_sr[sr_refpy:sr_refpy+4, sr_refpx:sr_refpx+4, : ]
                            #generate neural network input and label, use Ground truth of current frame and reference frame
                            gt_residual[(py*4):(py*4+4),(px*4):(px*4+4),:] = b_block_gt.astype("float") - ref.astype("float")
                        else:
                            gt_residual[(py*4):(py*4+4),(px*4):(px*4+4),:] = b_block_gt.astype("float")
            
            tmp_gt_residual = np.zeros((8, 8, 3))
            for px in range(int(4*block_w/8)):
                for py in range(int(4*block_h/8)):
                    # split the residual into 8*8 blocks
                    tmp_gt_residual = np.array(gt_residual[py*8:py*8+8,px*8:px*8+8,:])  
                    #generate high-resolution mv
                    tmp_curx = 4*curx + px*8
                    tmp_cury = 4*cury + py*8
                    tmp_refx = 4*refx + px*8
                    tmp_refy = 4*refy + py*8
                    #redefine macro-block as each block becomes 8*8
                    tmp_row = np.array(row).astype(int)
                    tmp_row[2] = int(8)
                    tmp_row[3] = int(8)
                    tmp_row[4] = tmp_curx # shift curx
                    tmp_row[5] = tmp_cury
                    tmp_row[6] = tmp_refx
                    tmp_row[7] = tmp_refy
                    test = [tmp_row,tmp_gt_residual]
                    overall_info.append(test)
                    
            
#note that P frame also be reconstructed
def DFS(fcnt):
    if vis[fcnt]:
        return True
    else:
        for i in mvsmat[fcnt]:
            DFS(i)
        bframe_gen_kernel(fcnt)
        
        vis[fcnt] = True
        return True

def bframe_gen():

    for i in bflist:
        # print(i)
        if not vis[i]:
            DFS(i)

    for i in range(frame_num):
        if not vis[i]:
            print("ERROR")


# reorder the sequence of decoding frame
def ReorderDecFrame(para_mvGroup):

	orderDecFrame = []  #house the index after reoder
	idx = 0  			#index for tranversing orderDecFrame

	for row in para_mvGroup:
		
		referenceIdx = row[1]
		newFrameIdx  = row[0] #the new referenced frame
		
		# Identify whether the reference frame firstly coming out
		if int(referenceIdx) not in orderDecFrame:
			# print('The reference frame\'s ' + referenceIdx + ' not be decoded')
			orderDecFrame.insert(idx, int(referenceIdx))
			idx = idx + 1
			
		# Identify  whether the referenced frame firstly coming out
		if int(newFrameIdx) not in orderDecFrame:
			# print('The reference frame\'s ' + newFrameIdx + ' not be decoded')
			orderDecFrame.insert(idx, int(newFrameIdx))
			idx = idx + 1
		
	return orderDecFrame



# iterate all videos
for i in classname_list:
    #init global variables
    mvsmat = {} # 记录各帧depending关系的dict
    bflist = []  # aka b frame list
    pflist = []  # aka b frame list
    Res_data = [] # a list to store Residual_data
    MV_data = [] # a list to store MV
    overall_info = [] # a list to store all info, including MV and frequency
    classname = i
    print("classname: ", classname)
    video_path = PICS_DIR + classname
    pic_names = os.listdir(video_path)
    frame_num = len(pic_names)
    vis = [False] * frame_num
    #get shape and length of the whole video
    img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = img.shape
    # print(frame_h, frame_w) #(120, 180)
    sr_frame_h = 4 * frame_h
    sr_frame_w = 4 * frame_w
    frame_mat_GT_HR = np.zeros((frame_num+1,sr_frame_h,sr_frame_w, 3), dtype="uint8") #init frame_mat
    frame_mat_GT_LR = np.zeros((frame_num+1,frame_h,frame_w, 3), dtype="uint8")
    frame_mat_SR_HR = np.zeros((frame_num+1,sr_frame_h,sr_frame_w, 3), dtype="uint8")

    #begin function
    she = shelve.open(TRAIN_DIR+"residual_"+classname+".bat")
    fetch_MV_data()
    dep_tree_gen()

    #generate depending order
    mvName_file = MVS_DIR+classname+".csv"
    csvMvFile = open(mvName_file)
    csvMvReader = csv.reader(csvMvFile)
    mvGroup = list(csvMvReader)	
    resultReoderIndex = ReorderDecFrame(mvGroup) # a list for storing decoding order
    for j in resultReoderIndex:
        if j in bflist: #only reconstruct B frame
            print("this is a B frame, the index is: ",j)
            bframe_gen_kernel(j)
        else:
            print("this is a I/P frame, the index is: ",j)
    she[classname] = overall_info # update shelve
    she.close() 
