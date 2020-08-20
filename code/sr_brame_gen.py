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


#整帧地进行SR

IDX_DIR="/home/songzhuoran/video/video-sr-acc/Info_BIx4/idx/"
MVS_DIR="/home/songzhuoran/video/video-sr-acc/Info_BIx4/mvs/"
B_DIR="/home/songzhuoran/video/video-sr-acc/bframe_sr_test/"
RES_DIR="/home/songzhuoran/video/video-sr-acc/Info_BIx4/Residuals/"
PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/BIx4/"
SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/EDVR/results/Vid4/"

mvsmat = {} # 记录各帧depending关系的dict
bflist = []  # aka b frame list
pflist = []  # aka b frame list
Res_data = [] # a list to store Residual_data
classname = 'calendar'
print('classname: ', classname)
video_path = '/home/songzhuoran/video/video-sr-acc/Vid4/BIx4/' + classname
pic_names = os.listdir(video_path)
frame_num = len(pic_names)
vis = [False] * frame_num

#get shape and length of the whole video
img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
frame_h, frame_w, _ = img.shape
# print(frame_h, frame_w) #(120, 180)
sr_frame_h = 4 * frame_h
sr_frame_w = 4 * frame_w
frame_mat = np.zeros((frame_num,sr_frame_h,sr_frame_w, 3), dtype="uint8")
# print(frame_mat, frame_mat.shape)

def fetch_res_data():
    #load the whole res_data file into a py list. prevent redundant loading
    with open(RES_DIR + "Bix4_res_"+classname + ".csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            Res_data.append(item)
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
        # print(pflist)
    for i in pflist:
        vis[i] = True
        f_num = '%08d' % i
        curstr = SR_PICS_DIR + classname + "/" + f_num + ".png"
        frame_mat[i] = cv2.imread(curstr,-1) #init frame
    for i in range(frame_num):
        mvsmat[i] = set()  #记录每一帧的还原需要哪些帧的数据
    with open(MVS_DIR+classname+".csv","r") as file:
        datainfo = csv.reader(file)
        for row in datainfo:
            mvsmat[int(row[0])].add(int(row[1]))
    # for i in mvsmat:
    #     if i in bflist:
    #         print(i, mvsmat[i])
    return

def res_ext(res_3d):
    #使用上采样将residual扩张为SR后图片的大小
    block_h, block_w, _ = res_3d.shape
    res_3d_ext = np.zeros((4*block_h, 4*block_w, 3))
    for i in range(3):

        res_2d = res_3d[:, :, i]
        y, x = np.mgrid[-1: 1: block_h*1j, -1: 1: block_w*1j]
        xnew = np.linspace(-1, 1, 4*block_w)
        ynew = np.linspace(-1, 1, 4*block_h)
        intp_func = interpolate.interp2d(x, y, res_2d, kind='cubic')
        res_2d_ext = intp_func(xnew, ynew)
        res_3d_ext[:, :, i] = res_2d_ext
    return res_3d_ext

def res_ext2(res_3d):
    #使用最邻近插值法
    block_h, block_w, _ = res_3d.shape
    res_3d_ext = np.zeros((4*block_h, 4*block_w, 3))
    for i in range(3):
        for py_ext in range(4*block_h):
            for px_ext in range(4 * block_w):
                py_org = min(round(py_ext / 4), 7)
                px_org = min(round(px_ext / 4), 7)
                res_3d_ext[py_ext,px_ext,i] = res_3d[py_org,px_org,i]
    return res_3d_ext


def bframe_gen_kernel1(fcnt):
    '''
        加入残差，将单个点直接复制16次进行插值
        ref图像为sr图像
    '''
    global Res_data  #函数内使用全局变量global
    print(fcnt, len(Res_data))
    new_Res_data = Res_data.copy()
    img_vis = np.zeros((frame_h, frame_w))
    bframe_img_sr = np.zeros((sr_frame_h, sr_frame_w, 3))
    for row in Res_data:
        if int(float(row[0])) == fcnt:
            new_Res_data.remove(row)  #每次减少Res_data数目，加速整体过程
            ref_frame_sr = frame_mat[int(float(row[1]))]
            ref_frame_sr = cv2.imread(SR_PICS_DIR+classname+"/%08d.png" % int(float(row[1])))
            # print('ref_frame', ref_frame_sr.shape)
            block_w, block_h, curx, cury, refx, refy = np.array(row[2:8]).astype(float).astype(int)
            res_1d = np.array(row[8:]).astype(float).astype(int)
            res_3d = res_1d.reshape((block_h, block_w, 3))
            for px in range(block_w):
               for py in range(block_h):
                    if (curx+px in range(frame_w)) and (cury+py in range(0, frame_h)):
                        # cur块在范围内
                        sr_curpx = 4 * (curx + px) # sr后图片内各点对应的坐标
                        sr_curpy = 4 * (cury + py)
                        sr_refpx = 4 * (refx + px)
                        sr_refpy = 4 * (refy + py)
                        ref = ref_frame_sr[sr_refpy:sr_refpy+4, sr_refpx:sr_refpx+4] \
                            if (refx+px in range(0, frame_w)) and (refy+py in range(0, frame_h)) \
                            else 0

                        res_RGB = res_3d[py,px]
                        res_ext = np.ones((4, 4, 3))
                        for i in range(3):
                            res_ext[:,:,i] = res_RGB[i] * np.ones((4,4))
                        bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                                      ref + res_ext
                        # print('ref', ref.shape)
                        #cur块在范围内
                        # if img_vis[cury + py][curx + px] == 0:
                        #     # 该块只有一个ref块
                        #     bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4] = \
                        #         ref + res_3d_ext[4*py:4*py+4,4*px:4*px+4]
                        #     # print(bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4])
                        #     # print('ref: ', ref.shape, ' res_3d:', res_3d_ext[4*py:4*py+4,4*px:4*px+4].shape)
                        # else:
                        #     # 该块有两个ref块
                        #     # print('hihihi  ', img_vis[cury + py][curx + px])
                        #     bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                        #         (ref + res_3d_ext[4*py:4*py+4, 4*px:4*px+4] +
                        #          bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4]) / 2
                        # img_vis[cury+py][curx+px] += 1   # 访问次数+1
                    else:
                        continue
    cv2.imwrite(B_TEST_DIR+classname+"/%08d.png" % fcnt, bframe_img_sr)
    Res_data = new_Res_data
    #

def bframe_gen_kernel2(fcnt):
    '''
        加入残差，使用interp函数进行线性插值
        ref图像为sr图像
    '''
    global Res_data  #函数内使用全局变量global
    print(fcnt, len(Res_data))
    new_Res_data = Res_data.copy()
    img_vis = np.zeros((frame_h, frame_w))
    bframe_img_sr = np.zeros((sr_frame_h, sr_frame_w, 3))
    for row in Res_data:
        if int(float(row[0])) == fcnt:
            new_Res_data.remove(row)  #每次减少Res_data数目，加速整体过程
            ref_frame_sr = frame_mat[int(float(row[1]))]
            ref_frame_sr = cv2.imread(SR_PICS_DIR+classname+"/%08d.png" % int(float(row[1])))
            # print('ref_frame', ref_frame_sr.shape)
            block_w, block_h, curx, cury, refx, refy = np.array(row[2:8]).astype(float).astype(int)
            res_1d = np.array(row[8:]).astype(float).astype(int)
            res_3d = res_1d.reshape((block_h, block_w, 3))
            res_3d_ext = res_ext(res_3d)  #对res进行插值
            for px in range(block_w):
               for py in range(block_h):
                    if (curx+px in range(frame_w)) and (cury+py in range(0, frame_h)):
                        # cur块在范围内
                        sr_curpx = 4 * (curx + px) # sr后图片内各点对应的坐标
                        sr_curpy = 4 * (cury + py)
                        sr_refpx = 4 * (refx + px)
                        sr_refpy = 4 * (refy + py)
                        ref = ref_frame_sr[sr_refpy:sr_refpy+4, sr_refpx:sr_refpx+4] \
                            if (refx+px in range(0, frame_w)) and (refy+py in range(0, frame_h)) \
                            else 0
                        bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                                      ref \
                                      + res_3d_ext[4*py:4*py+4,4*px:4*px+4]
                        # print('ref', ref.shape)
                        #cur块在范围内
                        # if img_vis[cury + py][curx + px] == 0:
                        #     # 该块只有一个ref块
                        #     bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4] = \
                        #         ref + res_3d_ext[4*py:4*py+4,4*px:4*px+4]
                        #     # print(bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4])
                        #     # print('ref: ', ref.shape, ' res_3d:', res_3d_ext[4*py:4*py+4,4*px:4*px+4].shape)
                        # else:
                        #     # 该块有两个ref块
                        #     # print('hihihi  ', img_vis[cury + py][curx + px])
                        #     bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                        #         (ref + res_3d_ext[4*py:4*py+4, 4*px:4*px+4] +
                        #          bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4]) / 2
                        # img_vis[cury+py][curx+px] += 1   # 访问次数+1
                    else:
                        continue
    cv2.imwrite(B_TEST_DIR+classname+"/%08d.png" % fcnt, bframe_img_sr)
    Res_data = new_Res_data
    #

def bframe_gen_kernel3(fcnt):
    '''
        加入残差，使用最近邻插值法进行插值
        ref图像为sr图像
    '''
    global Res_data  #函数内使用全局变量global
    print(fcnt, len(Res_data))
    new_Res_data = Res_data.copy()
    img_vis = np.zeros((frame_h, frame_w))
    bframe_img_sr = np.zeros((sr_frame_h, sr_frame_w, 3))
    for row in Res_data:
        if int(float(row[0])) == fcnt:
            new_Res_data.remove(row)  #每次减少Res_data数目，加速整体过程
            ref_frame_sr = frame_mat[int(float(row[1]))]
            ref_frame_sr = cv2.imread(SR_PICS_DIR+classname+"/%08d.png" % int(float(row[1])))
            # print('ref_frame', ref_frame_sr.shape)
            block_w, block_h, curx, cury, refx, refy = np.array(row[2:8]).astype(float).astype(int)
            res_1d = np.array(row[8:]).astype(float).astype(int)
            res_3d = res_1d.reshape((block_h, block_w, 3))
            res_3d_ext = res_ext2(res_3d)  #对res进行插值
            for px in range(block_w):
               for py in range(block_h):
                    if (curx+px in range(frame_w)) and (cury+py in range(0, frame_h)):
                        # cur块在范围内
                        sr_curpx = 4 * (curx + px) # sr后图片内各点对应的坐标
                        sr_curpy = 4 * (cury + py)
                        sr_refpx = 4 * (refx + px)
                        sr_refpy = 4 * (refy + py)
                        ref = ref_frame_sr[sr_refpy:sr_refpy+4, sr_refpx:sr_refpx+4] \
                            if (refx+px in range(0, frame_w)) and (refy+py in range(0, frame_h)) \
                            else 0
                        bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                                      ref \
                                      + res_3d_ext[4*py:4*py+4,4*px:4*px+4]
                        # print('ref', ref.shape)
                        #cur块在范围内
                        # if img_vis[cury + py][curx + px] == 0:
                        #     # 该块只有一个ref块
                        #     bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4] = \
                        #         ref + res_3d_ext[4*py:4*py+4,4*px:4*px+4]
                        #     # print(bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4])
                        #     # print('ref: ', ref.shape, ' res_3d:', res_3d_ext[4*py:4*py+4,4*px:4*px+4].shape)
                        # else:
                        #     # 该块有两个ref块
                        #     # print('hihihi  ', img_vis[cury + py][curx + px])
                        #     bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                        #         (ref + res_3d_ext[4*py:4*py+4, 4*px:4*px+4] +
                        #          bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4]) / 2
                        # img_vis[cury+py][curx+px] += 1   # 访问次数+1
                    else:
                        continue
    cv2.imwrite(B_TEST_DIR+classname+"/%08d.png" % fcnt, bframe_img_sr)
    Res_data = new_Res_data
    #

def bframe_gen_kernel4(fcnt):
    '''
        不加入残差，只使用mv
    '''
    global Res_data  #函数内使用全局变量global
    print(fcnt, len(Res_data))
    new_Res_data = Res_data.copy()
    img_vis = np.zeros((frame_h, frame_w))
    bframe_img_sr = np.zeros((sr_frame_h, sr_frame_w, 3))
    for row in Res_data:
        if int(float(row[0])) == fcnt:
            new_Res_data.remove(row)  #每次减少Res_data数目，加速整体过程
            ref_frame_sr = frame_mat[int(float(row[1]))]
            ref_frame_sr = cv2.imread(SR_PICS_DIR + classname + "/%08d.png" % int(float(row[1])))
            # print('ref_frame', ref_frame_sr.shape)
            block_w, block_h, curx, cury, refx, refy = np.array(row[2:8]).astype(float).astype(int)
            for px in range(block_w):
               for py in range(block_h):
                    if (curx+px in range(frame_w)) and (cury+py in range(0, frame_h)):
                        # cur块在范围内
                        sr_curpx = 4 * (curx + px) # sr后图片内各点对应的坐标
                        sr_curpy = 4 * (cury + py)
                        sr_refpx = 4 * (refx + px)
                        sr_refpy = 4 * (refy + py)
                        ref = ref_frame_sr[sr_refpy:sr_refpy+4, sr_refpx:sr_refpx+4] \
                            if (refx+px in range(0, frame_w)) and (refy+py in range(0, frame_h)) \
                            else 0
                        # bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                        #              ref
                        # print('ref', ref.shape)
                        #cur块在范围内
                        if img_vis[cury + py][curx + px] == 0:
                            # 该块只有一个ref块
                            bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4] = \
                                ref
                            # print(bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4])
                            # print('ref: ', ref.shape, ' res_3d:', res_3d_ext[4*py:4*py+4,4*px:4*px+4].shape)
                        else:
                            # 该块有两个ref块
                            # print('hihihi  ', img_vis[cury + py][curx + px])
                            bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                                (ref +
                                 bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4]) / 2
                        img_vis[cury+py][curx+px] += 1   # 访问次数+1
                    else:
                        continue
    cv2.imwrite(B_TEST_DIR+classname+"/%08d.png" % fcnt, bframe_img_sr)
    Res_data = new_Res_data
    #

def bframe_gen_kernel5(fcnt):
    '''
        加入残差，残差使用cv2库的resize函数进行插值(resize函数内可选择三种不同的插值方式)
    '''
    global Res_data  #函数内使用全局变量global
    print(fcnt, len(Res_data))
    new_Res_data = Res_data.copy()
    img_vis = np.zeros((frame_h, frame_w))
    bframe_img_sr = np.zeros((sr_frame_h, sr_frame_w, 3))
    for row in Res_data:
        if int(float(row[0])) == fcnt:
            new_Res_data.remove(row)  #每次减少Res_data数目，加速整体过程
            ref_frame_sr = frame_mat[int(float(row[1]))]
            # ref_frame_sr = cv2.imread(SR_PICS_DIR+classname+"/%08d.png" % int(float(row[1])))
            # print('ref_frame', ref_frame_sr.shape)
            block_w, block_h, curx, cury, refx, refy = np.array(row[2:8]).astype(float).astype(int)
            res_1d = np.array(row[8:]).astype(float).astype(np.int16)
            res_3d = res_1d.reshape((block_h, block_w, 3))
            res_3d_ext = cv2.resize(res_3d, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            # print(res_3d_ext.shape)
            for px in range(block_w):
               for py in range(block_h):
                    if (curx+px in range(frame_w)) and (cury+py in range(0, frame_h)):
                        # cur块在范围内
                        sr_curpx = 4 * (curx + px) # sr后图片内各点对应的坐标
                        sr_curpy = 4 * (cury + py)
                        sr_refpx = 4 * (refx + px)
                        sr_refpy = 4 * (refy + py)
                        ref = ref_frame_sr[sr_refpy:sr_refpy+4, sr_refpx:sr_refpx+4] \
                            if (refx+px in range(0, frame_w)) and (refy+py in range(0, frame_h)) \
                            else 0
                        bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                                      ref \
                                      + res_3d_ext[4*py:4*py+4,4*px:4*px+4]
                        # print('ref', ref.shape)
                        #cur块在范围内
                        # if img_vis[cury + py][curx + px] == 0:
                        #     # 该块只有一个ref块
                        #     bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4] = \
                        #         ref + res_3d_ext[4*py:4*py+4,4*px:4*px+4]
                        #     # print(bframe_img_sr[sr_curpy:sr_curpy+4,sr_curpx:sr_curpx+4])
                        #     # print('ref: ', ref.shape, ' res_3d:', res_3d_ext[4*py:4*py+4,4*px:4*px+4].shape)
                        # else:
                        #     # 该块有两个ref块
                        #     # print('hihihi  ', img_vis[cury + py][curx + px])
                        #     bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4] = \
                        #         (ref + res_3d_ext[4*py:4*py+4, 4*px:4*px+4] +
                        #          bframe_img_sr[sr_curpy:sr_curpy + 4, sr_curpx:sr_curpx + 4]) / 2
                        # img_vis[cury+py][curx+px] += 1   # 访问次数+1
                    else:
                        continue
    cv2.imwrite(B_DIR+classname+"/%08d.png" % fcnt, bframe_img_sr)
    Res_data = new_Res_data
    #


def DFS(fcnt):
    if vis[fcnt]:
        return True
    else:
        for i in mvsmat[fcnt]:
            DFS(i)
        bframe_gen_kernel5(fcnt)
        frame_mat[fcnt] = cv2.imread(B_DIR + classname + "/%08d.png" % fcnt)
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


fetch_res_data()
dep_tree_gen()
bframe_gen()
# bframe_gen_kernel3(2)