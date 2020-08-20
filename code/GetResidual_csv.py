#import torch
import cv2
import numpy as np
import glob
from PIL import Image
# import openpyxl
# import xlwt
import csv
import math


MV_DIR = "Info_BIx4/mvs/"
RES_DIR = "Info_BIx4/Residuals/"

Class = ["calendar", "city", "foliage", "walk"]
W = [180, 176, 180, 180]
H = [144, 144, 120, 120]
Frame = [41, 34, 49, 47]

# workbook = openpyxl.Workbook()
# worksheet = workbook.active
# worksheet.title='sheet1'
def get_resigual(ID):
	Datapath = "Vid4/BIx4/" + Class[ID] + "/*.png"
	w = W[ID]
	h = H[ID]
	frame = Frame[ID]
	imgs = np.zeros((frame, h, w, 3))
	i = 0
	# print(glob.glob(Datapath))
	for imageFile in sorted(glob.glob(Datapath)):
		# img = np.array(Image.open(imageFile))
		img = cv2.imread(imageFile)
		# print(img)
		imgs[i] = img
		i += 1

	#the whole mv file
	mv = np.loadtxt(open(MV_DIR + Class[ID] + ".csv","rb"),delimiter=",",skiprows=0).astype(int)
	result = [[]]*mv.shape[0]
	f = open(RES_DIR + 'Bix4_res_'+Class[ID] + '.csv','w',newline='')
	writer = csv.writer(f)
	for num in range(0,mv.shape[0]):
	# for num in range(1000):
		# print(num) if num%1000 = 0
		# print(num)
		cur = imgs[mv[num, 0]]
		ref = imgs[mv[num, 1]]
		block_w, block_h = mv[num, 2], mv[num, 3]
		block_res = np.zeros((block_h, block_w, 3))  # RGB 3 channels
		# print(block_w, block_h)
	# iterate over the whole macro block
		curx, cury, refx, refy = mv[num, 4:8]
		for sy in range(0, block_h):
			for sx in range(0, block_w):
				if cury+sy in range(0, h) and curx+sx in range(0, w):
					#cur块在范围内
					if refx+sx in range(0, w) and refy+sy in range(0, h):
						#cur块和ref块都在范围内
						mb_cur = cur[cury + sy, curx + sx]
						mb_ref = ref[refy + sy, refx + sx]
						diff = mb_cur - mb_ref
						block_res[sy, sx] = diff
					else:
						#cur块在范围内，ref块不在范围内
						mb_cur = cur[cury + sy, curx + sx]
						diff = mb_cur
						block_res[sy, sx] = diff
					# print(cur[cury + sy, curx + sx], ref[refy + sy, refx + sx],
					# 	  block_res[sy, sx],
					# 	  cur[cury + sy, curx + sx]==ref[refy + sy, refx + sx]+block_res[sy, sx])
				else: 
					#cur块不在范围内
					block_res[sy, sx] = 0

		block_res_1d = block_res.reshape(-1)
		line = np.append(mv[num], block_res_1d)
		# print(line.shape)
		# for column, data in enumerate(line):
		# 	# print(column, data)
		# 	worksheet.cell(num+1, column+1, float(data))
		writer.writerow(line)
		# break
	f.close()

	# workbook.save('Residuals/Bix4_res_'+Class[ID] + '.xlsx')
	# np.savetxt('abs_res_' + Class[ID] + '.csv', result, delimiter = ',', fmt='%.04f')
	return

# get_resigual(0)
for ID in range(0, 4):
	print(ID)
	get_resigual(ID)
