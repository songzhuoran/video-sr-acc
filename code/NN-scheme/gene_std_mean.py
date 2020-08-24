#### code for generating the std and mean values of 8*8*3 blocks
import shelve
import numpy as np

# # for Vid4 dataset
# TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/train.bat"
# db = shelve.open(TRAIN_DIR)

# for REDS dataset
TRAIN_DIR = "/home/songzhuoran/video/video-sr-acc/train_info/train_REDS.bat"
db = shelve.open(TRAIN_DIR)
she = {}
for name in db.keys() :
    she[name] = db[name]

db.close()
overall_info = []
overall_info = she["walk"]

for i in range(0, len(overall_info), int(len(overall_info)/20)) :
    test = []
    test = overall_info[i]
    # print("input: ",test[1])
    # print("label: ",test[2])

listInput = []
listLabel = []
for name in she.keys() :
    overall_info = she[name]
    for blockInfo in overall_info :
        listInput.append(blockInfo[1])
        listLabel.append(blockInfo[2])

arrInput = np.array(listInput)
arrLabel = np.array(listLabel)

print("meanInput:")
meanInput = np.mean(arrInput, axis=(0,))
print(meanInput.reshape(-1).tolist())
print("stdInput: ")
stdInput = np.std(arrInput, axis=(0,))
print(stdInput.reshape(-1).tolist())
print("meanLabel: ")
meanLabel = np.mean(arrLabel, axis=(0,))
print(meanLabel.reshape(-1).tolist())
print("stdLabel: ")
stdLabel = np.std(arrLabel, axis=(0,))
print(stdLabel.reshape(-1).tolist())

