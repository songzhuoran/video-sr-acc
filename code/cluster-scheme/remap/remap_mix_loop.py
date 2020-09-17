
#import torch
import cv2
import numpy as np
import glob
from PIL import Image
# import openpyxl
# import xlwt
import csv
import math
import os
from progressbar import *
'''
	对·SR后的结果进行重匹配(remap), 匹配对象为GT与SR后图片，以使得还原结果更接近GT
'''


PICS_DIR = "/home/yuzhongkai/super_resolution/Vid4/BIx4/"
SR_PICS_DIR = "/home/yuzhongkai/super_resolution/EDVR/results/Vid4/"
GT_PICS_DIR = "/home/yuzhongkai/super_resolution/Vid4/GT/"
MVS_DIR="/home/yuzhongkai/super_resolution/Info_BIx4/mvs/"
IDX_DIR="/home/yuzhongkai/super_resolution/Info_BIx4/idx/"
OUT_DIR = "remap_result/"





def dep_tree_gen():
	for i in range(frame_num):
		mvsmat[i] = set()  # 记录每一帧的还原需要哪些帧的数据
	with open(MVS_DIR+classname+".csv","r") as file:
		datainfo = csv.reader(file)
		for row in datainfo:
			mvsmat[int(row[0])].add(int(row[1]))

	with open(IDX_DIR + "b/" + classname, "r") as file:
		for row in file:
			bflist.append(int(row) - 1)
	# for i in mvsmat:
	#     print(i, mvsmat[i])
	return


def remap():

	mvs = np.loadtxt(open(MVS_DIR + classname + ".csv", "rb"), delimiter=",", skiprows=0).astype(int)
	f = open(OUT_DIR + 'remap_' + classname + '_20mix.csv', 'w', newline='')
	writer = csv.writer(f)
	a = 0
	# print('num ', len(mvs))
	widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
			   ' ', ETA()]
	pbar = ProgressBar(widgets=widgets).start()
	for i, mv in enumerate(mvs):
		pbar.update(i/(len(mvs)-1)*100)
		# if a < 20000:
		# 	a += 1
		# 	continue
		# print(a)
		a+=1
		curId = mv[0]
		refId = mv[1]
		blockw, blockh, curx, cury, refx, refy = mv[2:8]
		srcurx = 4 * curx
		srcury = 4 * cury
		srrefx = 4 * refx
		srrefy = 4 * refy
		srBlockw = 4 * blockw
		srBlockh = 4 * blockh
		# print(curx, cury, refx, refy)
		refs = mvsmat[curId]
		# refs = [refId]
		curFrame = cv2.imread(GT_PICS_DIR + classname + "/%08d.png" % curId).astype(int)
		srFrameh, srFramew, _ = curFrame.shape

		curBlock = np.zeros((srBlockh, srBlockw, 3))
		partCurBlock = curFrame[srcury: srcury+srBlockh, srcurx:srcurx+srBlockw]
		act_h, act_w, _ = partCurBlock.shape
		curBlock[0:act_h, 0:act_w] = curFrame[srcury:srcury+act_h, srcurx:srcurx+act_w]

		bestRefId, bestRefFrame, bestRefBlock, bestRefxy = None, None, None, None
		bestDiff = float('Inf')
		# refs = [ref]

		for ref in refs: # 在所有相关帧里重新搜索最匹配的块
			if ref in bflist:
				refFrame = cv2.imread(GT_PICS_DIR+classname+"/%08d.png" % ref)
			else:
				refFrame = cv2.imread(SR_PICS_DIR + classname + "/%08d.png" % ref)
			sRange = 20
			#确定重匹配搜索范围
			ymin = srrefy - sRange
			ymax = srrefy + sRange
			xmin = srrefx - sRange
			xmax = srrefx + sRange

			for fy in range(ymin, ymax): # 在该范围内搜索最合适的块
				for fx in range(xmin, xmax):

					refBlock = np.zeros((srBlockh, srBlockw, 3))
					# 此处与0取max意在防止负索引引起错误
					partRefBlock = refFrame[max(fy, 0): max(fy+srBlockh, 0),
								   max(fx, 0): max(fx+srBlockw, 0)]
					# 得到实际frame和block重合部分大小
					act_h, act_w, _ = partRefBlock.shape

					# frame代表frame参考下的坐标， block代表macro block参考下的坐标
					frameXmin = max(0, fx)
					frameXmax = max(0, min(fx+srBlockw, srFramew)) # 与0max方式负索引导致错误
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
		line = [curId, bestRefId, srBlockw, srBlockh, srcurx, srcury, bestRefxy[0], bestRefxy[1]]
		# print(cur_block.shape, best_ref_block.shape, best_refxy, act_h, act_w)
		res_3d = curBlock - bestRefBlock
		res_1d = res_3d.reshape(-1)
		line.extend(res_1d)
		writer.writerow(line)
	pbar.finish()
	f.close()


classList = ['calendar', 'city', 'foliage', 'walk']
# classList = ['foliage', 'walk']
for classname in classList:
	print('remaping class: ', classname, '......')
	pic_names = os.listdir(PICS_DIR + classname)
	frame_num = len(pic_names)
	vis = [False] * frame_num

	# get shape and length of the whole video
	img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
	frame_h, frame_w, _ = img.shape

	# print(frame_h, frame_w) #(120, 180)
	frame_mat = np.zeros((frame_num, frame_h, frame_w, 3), dtype="uint8")
	# print(frame_mat, frame_mat.shape)

	bflist = []
	pflist = []
	mvsmat = {}
	dep_tree_gen()
	remap()
	print('\n')




