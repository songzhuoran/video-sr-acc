import os

DIR = "/home/songzhuoran/video/video-sr-acc/GOPRO/SR_result/"

videoname = os.listdir(DIR)
print(videoname)

for i in videoname:
    filename = os.listdir(DIR+i)
    tmp_list = []
    for j in filename:
        j = j.replace('.png','')
        tmp_list.append(j)
    tmp_list.sort()
    min_num = int(tmp_list[0])
    new_list = []
    for j in tmp_list:
        # print(j)
        hhh = int(j) - min_num
        hhh_str = "/%06d.png" % hhh
        picname = DIR+i+'/'+j+'.png'
        os.rename(picname,DIR+i+hhh_str)