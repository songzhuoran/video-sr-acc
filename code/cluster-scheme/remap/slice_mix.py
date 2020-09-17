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
from progressbar import *

IDX_DIR="/home/yuzhongkai/super_resolution/Info_BIx4/idx/"
REMAP_RES_DIR = "/home/yuzhongkai/super_resolution/function/remap/remap_result/"
PICS_DIR = "/home/yuzhongkai/super_resolution/Vid4/BIx4/"
HR_PICS_DIR = "/home/yuzhongkai/super_resolution/Vid4/GT/"
SR_PICS_DIR = "/home/yuzhongkai/super_resolution/EDVR/results/Vid4/"

SLICES_OUT_DIR = 'slice_result/mix/'

def fetch_res_data():
    #load the whole res_data file into a py list. prevent redundant loading
    with open(REMAP_RES_DIR + "new_remap_"+classname + "_20mix.csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            Res_data.append(item)

def fetch_bp_frame():
    global bflist, pflist
    with open(IDX_DIR+"b/"+classname, "r") as file:
        for row in file:
            bflist.append(int(row)-1)

    with open(IDX_DIR+"p/"+classname, "r") as file:
        for row in file:
            pflist.append(int(row)-1)

def genSliceOrder(res_data):
    sliceOrderList = []  # store reorder index
    idx = 0  # index for tranversing orderDecFrame
    for row in res_data:
        refIdx = int(float(row[1]))
        curIdx = int(float(row[0]))  # the new referenced frame
        # Identify whether the reference frame firstly coming out
        if int(refIdx) not in sliceOrderList:
            # print('The reference frame\'s ' + referenceIdx + ' not be decoded')
            sliceOrderList.insert(idx, int(refIdx))
            idx = idx + 1
        # Identify  whether the referenced frame firstly coming out
        if int(curIdx) not in sliceOrderList:
            # print('The reference frame\'s ' + newFrameIdx + ' not be decoded')
            sliceOrderList.insert(idx, int(curIdx))
            idx = idx + 1
    return sliceOrderList

def slice_kernel(fid):
    global overall_info
    for i, row in enumerate(Res_data):
        if int(float(row[0])) == fid:
            cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = np.array(row[0:8]).astype(float).astype(int)
            res_1d = np.array(row[8:]).astype(float).astype(np.int16)
            res_3d = res_1d.reshape((block_h, block_w, 3))
            for px in range(int(block_w / 8)):
                for py in range(int(block_h / 8)):
                    tmp_gt_residual = np.array(res_3d[py * 8:py * 8 + 8, px * 8:px * 8 + 8, :])
                    # generate high-resolution mv
                    tmp_curx = curx + px * 8
                    tmp_cury = cury + py * 8
                    tmp_refx = refx + px * 8
                    tmp_refy = refy + py * 8
                    tmp_row = np.array(row[0: 8]).astype(int)
                    tmp_row[2] = int(8)
                    tmp_row[3] = int(8)
                    tmp_row[4] = tmp_curx  # shift curx
                    tmp_row[5] = tmp_cury
                    tmp_row[6] = tmp_refx
                    tmp_row[7] = tmp_refy
                    slice = [tmp_row, tmp_gt_residual]
                    overall_info.append(slice)


classnameList = ['calendar', 'city', 'foliage', 'walk']
# iterate all videos
for classname in classnameList:
    Res_data = [] # a list to store Residual_data
    bflist = []
    pflist = []
    overall_info = [] # a list to store all info, including MV and frequency

    print("slicing: ", classname, '-----------------------------------------------------')

    fetch_res_data()
    sliceOrder = genSliceOrder(Res_data)
    # print(sliceOrder)
    fetch_bp_frame()
    she = shelve.open(SLICES_OUT_DIR+"residual_"+classname+".bat")
    for j in sliceOrder:
        if j in bflist: #only reconstruct B frame
            print(" B  frame, index : ",j)
            slice_kernel(j)
        else:
            print("I/P frame, index : ",j)
    #generate depending order
    she[classname] = overall_info # update shelve
    she.close()
    print(classname, 'finish-----------------------------------------------------------')
