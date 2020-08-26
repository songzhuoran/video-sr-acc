## code for training NN, including define NN model and generate dataset
## code for finetuning NN for REDS dataset
## TODO: we need to retrain NN model on REDS dataset
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

from NN_Util import *

#for Vid4 dataset
modelName = '/home/songzhuoran/video/video-sr-acc/train_info/Vid4_NN/model_fc3'

#for REDS dataset
modelName_REDS = '/home/songzhuoran/video/video-sr-acc/train_info/REDS_NN/model_reds'

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(192, 1536)
        self.fc2 = nn.Linear(1536, 3072)
        self.fc3 = nn.Linear(3072, 192)
        # self.fc4 = nn.Linear(3072, 192)
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
        db = shelve.open(self.data_path)
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
                if i!=0: # need to substract the previous index num
                    test = overall_info[index-self.total[i-1]] # tmp_row,tmp_tmp_input_residual,tmp_tmp_label_residual
                else:
                    test = overall_info[index] # tmp_row,tmp_tmp_input_residual,tmp_tmp_label_residual
                MV = test[0]
                input_frequency = inputPreprocess(test[1].reshape((-1,)))
                label_frequency = labelPreprocess(test[2].reshape((-1,)))
                return input_frequency,label_frequency


if __name__ == '__main__':
    # # original training on Vid4 dataset
    # classname_list = ['calendar','city','foliage','walk']
    # train_dataset = ProductDataset('/home/songzhuoran/video/video-sr-acc/train_info/train.bat',classname_list,train=True)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                             batch_size=2048,
    #                                             shuffle=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # net = MyNet().to(device)
    # net=torch.load(modelName).to(device)


    # # loss and optimization
    # criterion = nn.MSELoss().to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.0000001)

    # # train
    # epoch = 0
    # while (True) :
    #     epoch += 1
    #     i = 0
    #     for data in train_loader:
    #         i = i+1
    #         net.zero_grad()
    #         input_frequency, label_frequency = data
    #         input_frequency = input_frequency.to(device)
    #         label_frequency = label_frequency.to(device)
    #         # print(input_frequency)
    #         output = net.forward(input_frequency.float())
    #         train_loss = criterion(output, label_frequency.float())

    #         train_loss.backward()
    #         optimizer.step()
    #         if i%10==0:
    #             MSE = np.mean(np.power(labelPostprocess(output.cpu().data.numpy()) - labelPostprocess(label_frequency.cpu().data.numpy()), 2))
    #             print("%d,%d loss: = %f, MSE: = %f" % (epoch, i+1, train_loss.data,MSE))
        
    #     if epoch%20==0:
    #         torch.save(net, modelName)

    # finetuning on REDS dataset
    # classname_list = ['000','001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029']
    # train_dataset = ProductDataset('/home/songzhuoran/video/video-sr-acc/train_info/train_REDS.bat',classname_list,train=True)
    classname_list = ['000'] # need to modify!!!
    train_dataset = ProductDataset('/home/songzhuoran/video/video-sr-acc/train_info/REDS_NN/train_REDS_000.bat',classname_list,train=True) # need to modify!!!
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=2048,
                                                shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = MyNet().to(device)
    # net=torch.load(modelName).to(device)
    net=torch.load(modelName_REDS).to(device)

    # freeze the first and last layers
    net.fc1.weight.requires_grad = False
    net.fc1.bias.requires_grad = False
    # net.fc3.weight.requires_grad = False
    # net.fc3.bias.requires_grad = False

    # loss and optimization
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    # train
    epoch = 0
    while (True) :
        epoch += 1
        i = 0
        for data in train_loader:
            i = i+1
            net.zero_grad()
            input_frequency, label_frequency = data
            input_frequency = input_frequency.to(device)
            label_frequency = label_frequency.to(device)
            # print(input_frequency)
            output = net.forward(input_frequency.float())
            train_loss = criterion(output, label_frequency.float())

            train_loss.backward()
            optimizer.step()
            if i%10==0:
                MSE = np.mean(np.power(labelPostprocess(output.cpu().data.numpy()) - labelPostprocess(label_frequency.cpu().data.numpy()), 2))
                print("%d,%d loss: = %f, MSE: = %f" % (epoch, i+1, train_loss.data,MSE))
        
        if epoch%10==0:
            torch.save(net, modelName_REDS)



