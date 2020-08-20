## code for inferencing NN, including reconstructing B frames
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import json
import os
import sys
import csv
from PIL import Image
import shelve
import cv2

from NN_Util import *

modelName = '/home/songzhuoran/video/video-sr-acc/train_info/model_fc3'

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(192, 1536)
        self.fc2 = nn.Linear(1536, 3072)
        self.fc3 = nn.Linear(3072, 192)
        # self.fc4 = nn.Linear(1536, 192)
        # self.fc5 = nn.Linear(840, 192)


    def forward(self, x): # NN inference
        # x = x.view(x.size()[0], -1)
        # print(x)
        # x = x.flatten(start_dim=1)
        # print(x.shape)
        # x = F.sigmoid(x)

        # x = F.relu(self.fc1(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


class ProductDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,classname_list,train = True):
        self.train = train
        self.data_path = data_path
        self.total = [0,0,0,0]
        self.classname_list = classname_list
        db = shelve.open(self.data_path+"train.bat")
        self.she = {}
        for name in db.keys() :
            self.she[name] = db[name]
        db.close()
        overall_info = []
        for i in range(0,len(self.classname_list)):
            classname = self.classname_list[i]
            overall_info = self.she[classname]
            if i !=0:
                self.total[i] = self.total[i-1] + len(overall_info)
            else:
                self.total[i] = len(overall_info) # count each file item number, 8000, 16000, 32000, 40000

    def __len__(self):
        length = len(self.classname_list)
        total = self.total[length-1] # count the total line number
        return total

    def __getitem__(self,index):
        
        overall_info = []
        for i in range(0,len(self.classname_list)):
            classname = self.classname_list[i]
            overall_info = self.she[classname] # load MV and frequency info of each 8*8 block
            if index<self.total[i]:
                # print(index)
                if i!=0: # need to substract the previous index num
                    test = overall_info[index-self.total[i-1]] # tmp_row,tmp_tmp_input_residual,tmp_tmp_label_residual
                else:
                    test = overall_info[index] # tmp_row,tmp_tmp_input_residual,tmp_tmp_label_residual
                MV = test[0].reshape(-1,)
                # if self.train == True:
                input_frequency = inputPreprocess(test[1].reshape((-1,)))
                label_frequency = labelPreprocess(test[2].reshape((-1,)))
                return i,MV,input_frequency,label_frequency

# generate the list for reconstruction
depending_list = []
def gene_depending_order():
    global depending_list
    #push all video list into it



if __name__ == '__main__':
    classname_list = ['calendar','city','foliage','walk']

    # init for reconstruction
    bflist = []
    frame_mat_SR = []
    frame_mat_GT_HR = []
    frame_mat_GT_LR = []
    IDX_DIR="/home/songzhuoran/video/video-sr-acc/Info_BIx4/idx/"
    PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/BIx4/" # GT_LR_pic
    HR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/GT/" # GT_HR_pic
    SR_PICS_DIR = "/home/songzhuoran/video/video-sr-acc/EDVR/results/Vid4/"
    RESULT_DIR = "/home/songzhuoran/video/video-sr-acc/bframe_sr/"
    NN_INFO_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/"
    for classname in classname_list:
        # print(classname)
        with open(IDX_DIR+"b/"+classname, "r") as file:
            tmp_list = []
            for row in file:
                tmp_list.append(int(row)-1)
            bflist.append(tmp_list) #bflist[0] is the list of calendar
            # print(tmp_list)
            video_path = PICS_DIR + classname
            pic_names = os.listdir(video_path)
            frame_num = len(pic_names)
            img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_h, frame_w, _ = img.shape
            sr_frame_h = 4 * frame_h
            sr_frame_w = 4 * frame_w

            tmp_frame_mat_SR = np.zeros((frame_num,sr_frame_h,sr_frame_w, 3), dtype="uint8") #init frame_mat
            tmp_frame_mat_GT_HR = np.zeros((frame_num,sr_frame_h,sr_frame_w, 3), dtype="uint8")
            tmp_frame_mat_GT_LR = np.zeros((frame_num,frame_h,frame_w, 3), dtype="uint8")
            for i in range(frame_num):
                tmp_frame_mat_SR[i] = cv2.imread(SR_PICS_DIR + classname + "/%08d.png" % i) # read SR result
                tmp_frame_mat_SR[i]=cv2.cvtColor(tmp_frame_mat_SR[i], cv2.COLOR_BGR2RGB)
                tmp_frame_mat_GT_HR[i] = cv2.imread(HR_PICS_DIR + classname + "/%08d.png" % i) # read GT_HR result
                tmp_frame_mat_GT_HR[i]=cv2.cvtColor(tmp_frame_mat_GT_HR[i], cv2.COLOR_BGR2RGB)
                tmp_frame_mat_GT_LR[i] = cv2.imread(PICS_DIR + classname + "/%08d.png" % i) # read GT_LR result
                tmp_frame_mat_GT_LR[i]=cv2.cvtColor(tmp_frame_mat_GT_LR[i], cv2.COLOR_BGR2RGB) #convert to rgb
                if i in tmp_list: #only init B frame as red figure
                    tmp_img = np.stack([np.zeros((sr_frame_h,sr_frame_w),'int'), np.zeros((sr_frame_h,sr_frame_w),'int'), np.ones((sr_frame_h,sr_frame_w),'int')*255], axis=2)
                    cv2.imwrite(RESULT_DIR+classname+"/%08d.png" % i, tmp_img)
            frame_mat_SR.append(tmp_frame_mat_SR)
            frame_mat_GT_HR.append(tmp_frame_mat_GT_HR)
            frame_mat_GT_LR.append(tmp_frame_mat_GT_LR)
            # print(frame_mat_SR)
    # end init for reconstruction

    train_dataset = ProductDataset(NN_INFO_DIR,classname_list,train=False)

    #NN inference, batch size = 1, non shuffle
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=1,
                                                shuffle=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=torch.load(modelName).to(device)

    #init
    prev_idx = -1
    prev_classname = "000"
    prev_classnum = -1
    classname = classname_list[0]
    video_path = PICS_DIR + classname
    pic_names = os.listdir(video_path)
    img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = img.shape
    sr_frame_h = 4 * frame_h
    sr_frame_w = 4 * frame_w
    B_img = np.zeros((sr_frame_h,sr_frame_w, 3), dtype="uint8")


    for data in train_loader:
        class_num,MV,input_frequency,label_frequency = data
        cur_idx, ref_idx, block_w, block_h, curx, cury, refx, refy = MV.data.numpy()[0,:]

        classname = classname_list[class_num]

        # End of a frame, Save the frame
        if ((cur_idx != prev_idx) or (prev_classname != classname)) :
            if not ((prev_classnum == -1) or (prev_idx == -1))  :
                if (prev_idx in bflist[prev_classnum]) :
                    print("write back a B frame, classname = %s, index = %d, target = %s"%(prev_classname, prev_idx, RESULT_DIR+prev_classname+"/%08d.png" % prev_idx))
                    B_img=cv2.cvtColor(B_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(RESULT_DIR+prev_classname+"/%08d.png" % prev_idx, B_img)
        # Start of a class, update frame size
        if prev_classname != classname:
            video_path = PICS_DIR + classname
            pic_names = os.listdir(video_path)
            img = cv2.imread(PICS_DIR + classname + '/' + pic_names[0], -1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_h, frame_w, _ = img.shape
            sr_frame_h = 4 * frame_h
            sr_frame_w = 4 * frame_w
            print("sr_frame_h: ", sr_frame_h)
            print("sr_frame_w: ",sr_frame_w)

        # Start of a frame, Init the buffer
        if (cur_idx != prev_idx) :
            B_img = np.zeros((sr_frame_h,sr_frame_w, 3), dtype="uint8")
            if cur_idx in bflist[class_num]:
                print("New B frame!")
        
        prev_idx = cur_idx
        prev_classname = classname
        prev_classnum = class_num

        if cur_idx in bflist[class_num]: #only reconstruct B frame

            #NN inference
            input_frequency = input_frequency.to(device)
            output = net.forward(input_frequency.float())
            fre_res_block = labelPostprocess(output.cpu().data.numpy())
            fre_res_block = fre_res_block.reshape((8,8,3))
            # fre_res_block = input_frequency # need to modify!!!!
            # fre_res_block = fre_res_block.reshape((8,8,3))

            #idct
            tmp_fre_res_block = np.zeros((8, 8, 3))
            tmp_fre_res_block[:,:,0] = cv2.idct(fre_res_block[:,:,0].astype(np.float))
            tmp_fre_res_block[:,:,1] = cv2.idct(fre_res_block[:,:,1].astype(np.float))
            tmp_fre_res_block[:,:,2] = cv2.idct(fre_res_block[:,:,2].astype(np.float))

            #yuv convert to rgb
            tmp_tmp_fre_res_block = np.zeros((8, 8, 3))
            tmp_tmp_fre_res_block[:,:,0] = tmp_fre_res_block[:,:,0] + 1.4075 * tmp_fre_res_block[:,:,2]
            tmp_tmp_fre_res_block[:,:,1] = tmp_fre_res_block[:,:,0] - 0.3455 * tmp_fre_res_block[:,:,1] - 0.7169 * tmp_fre_res_block[:,:,2]
            tmp_tmp_fre_res_block[:,:,2] = tmp_fre_res_block[:,:,0] + 1.779 * tmp_fre_res_block[:,:,1]
            # print(tmp_tmp_fre_res_block)           
            
            #init 8*8 reference block
            output_block = np.zeros((8, 8, 3))
            if ref_idx in bflist[class_num]: # reference frame is B frame
                ref_frame_sr = cv2.imread(RESULT_DIR+classname+"/%08d.png" % ref_idx)
                ref_frame_sr=cv2.cvtColor(ref_frame_sr, cv2.COLOR_BGR2RGB)
                
            else: # reference frame is I/P frame
                ref_frame_sr = frame_mat_SR[class_num][ref_idx,:,:,:].copy()

            #generate the 8*8 block
            for px in range(8):
                for py in range(8):
                    sr_curpx = curx + px # sr后图片内各点对应的坐标
                    sr_curpy = cury + py
                    sr_refpx = refx + px
                    sr_refpy = refy + py
                    if (sr_curpx < sr_frame_w) and (sr_curpy < sr_frame_h): # cur块在范围内
                        if (sr_refpx < sr_frame_w) and (sr_refpy < sr_frame_h):
                            ref = ref_frame_sr[sr_refpy, sr_refpx, :]
                            output_block[py,px,:] = (ref.astype("float") + tmp_tmp_fre_res_block[py,px,:].astype("float")).astype("float")
                        else:
                            output_block[py,px,:] = tmp_tmp_fre_res_block[py,px,:].astype("float")

            # write back the 8*8 result
            for px in range(8):
                for py in range(8):
                    for pc in range(3):
                        x = curx + px
                        y = cury + py
                        if (x < sr_frame_w) and (y < sr_frame_h):
                            if output_block[py,px,pc] < 0:
                                output_block[py,px,pc] = 0
                            if output_block[py,px,pc] > 255:
                                output_block[py,px,pc] = 255
                            B_img[y,x,pc] = output_block[py,px,pc].astype("uint8")
            




