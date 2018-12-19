
import torch
import numpy as np
from torchvision import transforms, utils
import cv2
#Dataloading
import readDataFiles as readDataFiles



def getVideo(filename, curTotal, curCount, w, h):
    capture = cv2.VideoCapture('video/' + filename + '.mp4')
    while capture.isOpened():
        ret, cur_frame= capture.read()
        if ret:
            #Normalize values between 0.0-1.0
            cur_frame = np.divide(cur_frame.astype(np.float32), 255)

            #List manipulation, taking all previous index's as first parameter
            #using ..., then manipulating frames from BGR to RGB.
            cur_frame = cur_frame[...,[2,1,0]]

            #Covert frame to expected dimensions, consider loss of quality.
            resizedFrame = cv2.resize(cur_frame, (w, h))
            curCount += 1
            #print(type(frame), frame.shape)
            curTotal = np.add(curTotal, resizedFrame)
        else:
            break
    capture.release()
    return curTotal, curCount




desired_w, desired_h = 224, 224
total = np.zeros((desired_w, desired_h, 3))
print("\n~~~| getTVSumMeanImage.py Execution |~~~")
videoNames, scores = readDataFiles.readScoresCSV('shotScores.csv')
segments, fCount = readDataFiles.getSegments('videoData.csv')
count = 0

for vid in videoNames:
    total, count = getVideo(vid, total, count, desired_h, desired_w)
    print('done' + vid)

mean = np.divide(total, count)
meanTensor = torch.tensor(mean)
torch.save(mean, "mean.pt")
