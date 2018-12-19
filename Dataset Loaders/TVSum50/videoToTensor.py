"""
Michael Lombardo
HSA-RNN - Thesis
TVSum50 Data Augmentation
Video To Tensor
Create torch tensor files of each segment, save as .pt to be
loaded by custom TVSumDataset.
"""

import torch
import numpy as np
from torchvision import transforms, utils
import cv2
import matplotlib.pyplot as plt
import time
#Dataloading
import readDataFiles as readDataFiles

def main(width, height):
    print('\n~~~| videoToTensor.py Execution |~~~')
    segments, fCount = readDataFiles.getSegments('videoData.csv')
    desired_w, desired_h = width, height
    #distance = 10

    count = 0
    mean = (torch.load('mean.pt'))

    print("Creating Tensor Files.")
    for seg in segments:
        #Progress feedback.
        if (count % 50 == 0):
            print(str(count/len(segments)*100) + "%")
        #Using distance
        #currentVideo = getVideo(seg[0], seg[1], seg[2], distance)
        currentVideo = readDataFiles.getVideo(seg[0], seg[1], seg[2], desired_w, desired_h, mean)
        '''
        #Visualize output
        for i in range(30):
            plt.imshow(currentVideo[i,...])
            plt.show()
            print(i)
        break
        '''
        torch.save(currentVideo, "tensors/" + str(seg[0]) + str(seg[1]) + ".pt")
        count +=1
    print('~~~| videoToTensor.py Complete |~~~\n')
