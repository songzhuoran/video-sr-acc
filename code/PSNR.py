import os
import sys
from PIL import Image
import numpy as np
import copy
import cv2
import matplotlib
from matplotlib import pyplot as plt 

matplotlib.use('AGG')

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# RGB -> YCbCr
def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def PSNRSingleChannel(x, y, max = 255) :
    assert (x.shape == y.shape), "Size mismatch."
    epi = 0.0001

    #print("np.product(x.shape)",np.product(x.shape))
    mse = np.sum(np.power((x - y), 2)) / (np.product(x.shape)) + epi
    # print("mse",mse)
    return 10 * np.log(max * max / mse) / np.log(10)

def usageHalt() :
    print("Usage: PSNR.py folderOfGT folderOfOutput")
    exit()

def get_PSNR(classname):

    path1 = "/home/songzhuoran/video/video-sr-acc/REDS/Our_result/bframe_sr_reconstruction/" + classname  # folder of output
    #path1 = "/home/songzhuoran/video/video-sr-acc/REDS/SR_result/" + classname
    path2 = "/home/songzhuoran/video/video-sr-acc/REDS/GT/" + classname  # folder of ground truth
    # path1 = "/home/songzhuoran/video/video-sr-acc/Vid4/SR_result/" + classname # folder of EDVR results
    # path1 = "/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/bframe_sr_reconstruction/" + classname # folder of EDVR results
    # path1 = "/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/bframe_sr_mix/" + classname # folder of EDVR results
    # path2 = "/home/songzhuoran/video/video-sr-acc/Vid4/GT/" + classname  # folder of ground truth
    # path1 = "/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/bframe_sr_reconstruction/" + classname # folder of EDVR results
    # path1 = "/home/songzhuoran/video/video-sr-acc/Vid4/Info_GT_test/images/" + classname
    # path2 = "/home/songzhuoran/video/video-sr-acc/Vid4/GT/" + classname  # folder of ground truth


    # path2 = path1
    list1 = list(filter(os.path.isfile,map(lambda x: os.path.join(path1, x), os.listdir(path1))))
    list2 = list(filter(os.path.isfile,map(lambda x: os.path.join(path2, x), os.listdir(path2))))

    list1.sort()
    list2.sort()

    if (len(list1) != len(list2)) :
        print('Numbers of files contained in two folder is different.', (len(list1), len(list2)))
        if (len(list1) > len(list2)) :
            print('Too few image in target folder. Halt.')
            exit()

    listAns = []

    def arrayYielder(arr) :
        for i in range(arr.shape[-1]) :
            yield arr[..., i]

    for p1,p2 in zip(list1, list2) :
        # print('hi')
        
        img1 = Image.open(p1)
        # arr1 = rgb2ycbcr(np.array(img1, dtype = "double"))
        arr1 = (np.array(img1, dtype = "double"))
        img2 = Image.open(p2).resize(img1.size, Image.BICUBIC)
        # arr2 = rgb2ycbcr(np.array(img2, dtype = "double"))
        arr2 = (np.array(img2, dtype = "double"))

        # delta_arr = arr1 - arr2
        # print("mean: ", np.mean(delta_arr))
        # print("std: ", np.std(delta_arr))
        # print("max: ", np.max(delta_arr))
        # delta_arr_reshape = delta_arr.reshape(-1)
        # x = np.arange(0, delta_arr_reshape.shape[0])
        # # plt.plot(x, delta_arr_reshape) 
        # plt.clf()
        # plt.hist(delta_arr_reshape,bins = list(range(-50,51)))
        # plt.savefig('/home/songzhuoran/video/video-sr-acc/test/'+p1.split('/')[-1]+'.png')

        # # draw the PSNR hit map
        # DRAW_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/draw/"
        # tmp_p1 = p1.replace("/home/songzhuoran/video/video-sr-acc/Vid4/Our_result/bframe_sr_reconstruction/",DRAW_DIR)
        # draw_img = arr1
        # for i in range(arr1.shape[0]):
        #     for j in range(arr2.shape[1]):
        #         a = abs(arr1[i,j,:]-arr2[i,j,:])
        #         b = np.sum(a)
        #         Rmin = 0
        #         Rmax = 255
        #         Cmin = np.array([0, 0, 255])
        #         Cmax = np.array([255, 0, 0])
        #         draw_img[i,j,:] = (b - Rmin) / (Rmax - Rmin) * (Cmax - Cmin) + Cmin
        # Image.fromarray(draw_img.astype('uint8')).save(tmp_p1)
        # print(tmp_p1)

        # calculate PSNR
        listAns += [np.sum(list(map(lambda pair: PSNRSingleChannel(pair[0], pair[1]), zip(arrayYielder(arr1),arrayYielder(arr2))))) / 3]

    avg = np.average(listAns)

    print('classname: ', classname)
    print(listAns)
    print('avg: ', avg)
    return
#
# classes = ['calendar', 'city', 'foliage', 'walk']
# classes = ['city']
# classes = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029']
classes = ['024','025','026','027','028','029'] # need to modify!
for classname in classes:
    get_PSNR(classname)
# get_PSNR('calendar')
