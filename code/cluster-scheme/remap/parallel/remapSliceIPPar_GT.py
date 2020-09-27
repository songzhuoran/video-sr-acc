from progressbar import *
import shelve
import numpy as np
import cv2
import os
import sys
import warnings
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import Manager
import threading
import time
# warnings.filterwarnings("ignore")

IDX_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4/idx/"
PICS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/BIx4/"
GT_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/GT/"
SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/SR_result/"
MVS_DIR = "/home/songzhuoran/video/video-sr-acc/REDS/Info_BIx4/mvs/"

# IDX_DIR = "/home/songzhuoran/video/video-sr-acc/Sintel/Info_BIx4/idx/"
# PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Sintel/BIx4/"
# GT_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Sintel/GT/"
# SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Sintel/SR_result/"
# MVS_DIR = "/home/songzhuoran/video/video-sr-acc/Sintel/Info_BIx4/mvs/"

SLICES_OUT_DIR = '/home/songzhuoran/video/video-sr-acc/train_info/REDS_Cluster_remap/IP_GT_SR/'


def fetch_mv_data():
    '''
        将mv数据全部转存到list中
    '''
    global mvsmat, mvs_data
    for i in range(frame_num):
        mvsmat[i] = set()  # 记录每一帧的还原需要哪些帧的数据
    mvs_data = np.loadtxt(open(MVS_DIR + classname + ".csv", "rb"),
                     delimiter=",", skiprows=0).astype(int)
    for row in mvs_data:
        mvsmat[int(row[0])].add(int(row[1]))
    # for i in mvsmat:
    #     print(i, mvsmat[i])
    return


def fetch_bp_frame():
    '''
        获取b frames 和 i/p frames列表
    '''
    global bflist, pflist
    with open(IDX_DIR+"b/"+classname, "r") as file:
        for row in file:
            bflist.append(int(row)-1)

    with open(IDX_DIR+"p/"+classname, "r") as file:
        for row in file:
            pflist.append(int(row)-1)


def genSliceOrder(data):
    '''
        得到reconstruct顺序列表
    '''
    sliceOrderList = []  # store reorder index
    idx = 0  # index for tranversing orderDecFrame
    for row in data:
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


def remapSliceKernel(fidx, mvs_data, bflist, dic):
    '''
        对不同帧进行并行remap+slice的子进程
    '''
    sliceList = [] # 储存该帧所有slice出的8x8块
    for i, row in enumerate(mvs_data):
        # print('generating frame: ', fidx, 'itering line: ', i, end='\n')
        # print('itering line: ', i, end='\r')
        if int(float(row[0])) != fidx:
            continue
        curId, refId = row[0: 2]
        blockw, blockh, curx, cury, refx, refy = row[2:8]
        srcurx, srcury, srrefx, srrefy = 4*curx, 4*cury, 4*refx, 4*refy
        srBlockw, srBlockh = 4 * blockw, 4 * blockh
        # curFrame = cv2.imread(
        #     GT_PICS_DIR + classname + "/frame_%04d.png" % (curId+1)) # add by songzhuoran, Sintel dataset
        curFrame = cv2.imread(
            GT_PICS_DIR + classname + "/%08d.png" % curId) # add by songzhuoran
        curFrame=cv2.cvtColor(curFrame, cv2.COLOR_BGR2RGB) # add by songzhuoran
        curFrame = curFrame.astype('float') # add by songzhuoran
        srFrameh, srFramew, _ = curFrame.shape
        curBlock = np.zeros((srBlockh, srBlockw, 3))
        partCurBlock = curFrame[srcury: srcury +
                                srBlockh, srcurx:srcurx+srBlockw] #curblock在帧内重合部分
        act_h, act_w, _ = partCurBlock.shape
        curBlock[0:act_h, 0:act_w] = curFrame[srcury:srcury +
                                                act_h, srcurx:srcurx+act_w]
        bestRefId, bestRefBlock, bestRefxy = None, None, None
        bestDiff = float('Inf')
        # refs = mvsmat[curId]
        # refs = [refId]
        refs = [f for f in mvsmat[curId] if f not in bflist]
        # print(refs)
        for ref in refs:  # 在所有相关帧里重新搜索最匹配的块
            # refFrame = cv2.imread(SR_PICS_DIR+classname+"/frame_%04d.png" % (ref+1)) # add by songzhuoran, sintel dataset
            refFrame = cv2.imread(SR_PICS_DIR+classname+"/%08d.png" % ref) # add by songzhuoran
            refFrame=cv2.cvtColor(refFrame, cv2.COLOR_BGR2RGB) # add by songzhuoran
            refFrame = refFrame.astype('float') # add by songzhuoran
            sRange = 20
            # 确定remap的范围
            ymin = srrefy - sRange
            ymax = srrefy + sRange
            xmin = srrefx - sRange
            xmax = srrefx + sRange
            for fy in range(ymin, ymax):  # 在该范围内搜索最合适的块
                for fx in range(xmin, xmax):
                    refBlock = np.zeros((srBlockh, srBlockw, 3))
                    # 此处与0取max意在防止负索引引起错误
                    partRefBlock = refFrame[max(fy, 0): max(fy+srBlockh, 0),
                                            max(fx, 0): max(fx+srBlockw, 0)]
                    # 得到实际frame和block重合部分大小
                    act_h, act_w, _ = partRefBlock.shape

                    # frame代表frame参考下的坐标， block代表macro block参考下的坐标
                    frameXmin = max(0, fx)
                    # 与0max方式负索引导致错误
                    frameXmax = max(0, min(fx+srBlockw, srFramew))
                    blockXmin = frameXmin - fx
                    blockXmax = blockXmin + act_w
                    frameYmin = max(0, fy)
                    frameYmax = max(0, min(fy+srBlockh, srFrameh))
                    blockYmin = frameYmin - fy
                    blockYmax = blockYmin + act_h
                    refBlock[blockYmin: blockYmax, blockXmin: blockXmax] = \
                        refFrame[frameYmin: frameYmax, frameXmin:frameXmax]
                    diff = np.sum((curBlock - refBlock) ** 2)
                    if diff < bestDiff:
                        bestRefId = ref
                        bestRefBlock = refBlock
                        bestRefxy = (fx, fy)
                        bestDiff = diff

        res_3d = (curBlock - bestRefBlock).astype(np.int16) # 使用int16压缩文件大小
        # res_3d = (curBlock - bestRefBlock)

        #begin to slice res data
        for px in range(int(srBlockw / 8)):
            for py in range(int(srBlockh / 8)):
                tmp_gt_residual = np.array(
                    res_3d[py * 8:py * 8 + 8, px * 8:px * 8 + 8, :])
                # print(type(tmp_gt_residual[0,0,0]))
                # generate high-resolution mv
                tmp_curx = srcurx + px * 8
                tmp_cury = srcury + py * 8
                tmp_refx = bestRefxy[0] + px * 8
                tmp_refy = bestRefxy[1] + py * 8
                tmp_row = np.array(row[0: 8]).astype(int)
                tmp_row[0] = curId
                tmp_row[1] = bestRefId
                tmp_row[2] = int(8)
                tmp_row[3] = int(8)
                tmp_row[4] = tmp_curx  # shift curx
                tmp_row[5] = tmp_cury
                tmp_row[6] = tmp_refx
                tmp_row[7] = tmp_refy
                slice = [tmp_row, tmp_gt_residual]
                sliceList.append(slice)
    dic[fidx] = sliceList
    print('finish: ', fidx)

# classnameList = ['calendar', 'city', 'foliage', 'walk']
# classnameList = ['city']
classnameList = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029']
# classnameList = ['ambush_1', 'market_4', 'temple_1', 'mountain_2', 'bamboo_3', 'wall', 'market_1', 'PERTURBED_market_3', 'cave_3', 'PERTURBED_shaman_1', 'ambush_3', 'tiger']

# iterate all videos
for classname in classnameList:
    mvsmat = {}
    mvs_data = []  # a list to store Residual_data
    bflist = []
    pflist = []
    pic_names = os.listdir(PICS_DIR + classname)
    frame_num = len(pic_names)
    print("slicing: ", classname,
          '-----------------------------------------------------')
    fetch_mv_data()
    sliceOrder = genSliceOrder(mvs_data)
    fetch_bp_frame()
    bframeGenList = [i for i in sliceOrder if i in bflist]
    totalBframes = len(bframeGenList)
    print('remaping %d bframes in the following order: \n' % totalBframes, bframeGenList, '\n')
    
    # begin parallel computing （remaping and slicing）
    start_time = time.time()
    dic = Manager().dict()
    shareMvs = Manager().list(mvs_data)
    childList = []
    for i, fidx in enumerate(bframeGenList):
        p = Process(target=remapSliceKernel, args=(fidx, shareMvs, bflist, dic))
        childList.append(p)
        p.start()
        print('generating %s dataset, frame: ' % classname, fidx)
    for p in childList:
        p.join()
    # parallel computing finished

    # reorder slicing result and store in shelve file
    overall_info = []  # a list to store all info, including MV and frequency
    she = shelve.open(SLICES_OUT_DIR+"residual_"+classname+".bat")
    for f in bframeGenList:
        overall_info.extend(dic[f])
    she[classname] = overall_info  # update shelve
    she.close()

    print('total_time: ', time.time()-start_time)
    print(classname, 'finish-----------------------------------------------------------')
