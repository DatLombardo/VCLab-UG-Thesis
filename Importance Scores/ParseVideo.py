"""
Michael Lombardo
Thesis
ParseVideo

Execution is just simply running the program, generates tensor of
each video in shotScores.csv. Each video is parsed to remove dark, blurry, and
uniform frames using low-level features, then based on defined CF value
(cut frames) will take every CF'th frame such that possibly every 5th or 10th
frame is taken from the video. Lastly, the CF video sequence is passed through
VGG16 (Pretrained) to extract frame level features.

Presets are: cf: 5, width:224, height:224.
"""
import random
import cv2
import numpy as np
import torch
import torchvision.models as models
import csv
import matplotlib.pyplot as plt
import time
import readDataFiles as readDataFiles

def getMean(videoData, width, height):
    """
    TO BE COMPLETED
    """
    curTotal = np.zeros((width, height, 3))
    curCount = 0
    for elem in videoData:
        if (curCount % 50 == 0):
            print(str(curCount/len(videoData)*10) + "%")

        cap = cv2.VideoCapture("video/"+elem[0]+".mp4")
        startOne = elem[1]
        count = 0

        while cap.isOpened():
            ret, cur_frame = cap.read()
            if ret:
                #Normalize values between 0.0-1.0
                cur_frame = np.divide(cur_frame.astype(np.float32), 255)

                #List manipulation, taking all previous index's as first parameter
                #using ..., then manipulating frames from BGR to RGB.
                cur_frame = cur_frame[...,[2,1,0]]

                #Covert frame to expected dimensions, consider loss of quality.
                resizedFrame = cv2.resize(cur_frame, (width, height))
                curCount += 1
                count += 1
                if count == 9:
                    break
                #print(type(frame), frame.shape)
                curTotal = np.add(curTotal, resizedFrame)
            else:
                break
        cap.release()

    return curTotal, curCount


def removeBadFrames(video, scores, w, h):
    cap = cv2.VideoCapture("video/"+video+".mp4")
    count = 0
    delCount = 0
    newVid = []
    while cap.isOpened():
        ret, currFrame = cap.read()
        if ret:
            currFrame = cv2.resize(currFrame, (w, h))[...,[2,1,0]]
            if readDataFiles.darkLabel(currFrame):
                #print(count, 'Dark')
                del scores[count-delCount]
                delCount +=1
            elif readDataFiles.blurryLabel(currFrame):
                #print(count, 'Blurry')
                del scores[count-delCount]
                delCount += 1
            elif readDataFiles.uniformLabel(currFrame):
                #print(count, 'Uniform')
                del scores[count-delCount]
                delCount += 1
            else:
                newVid.append(currFrame)
            count += 1
        else:
            break

    cap.release()
    if not(len(newVid) == len(scores)):
        scores.pop()

    return newVid, scores

def parseCF(video, scores, cf):
    #Take every cf'th element from the list
    print(len(video))
    print(len(video[::cf]))
    return [video[::cf], scores[::cf]]


def main():
    desired_w, desired_h = 224, 224

    print("~~ Generating Video Tensors ~~")

    vgg16 = models.vgg16(pretrained=True).cuda()
    for param in vgg16.parameters():
        param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
    vgg16_fcn = vgg16.features.cuda()
    vgg16_fcn.cuda()

    #Reads in list of potential videos to make custom segments with.
    vidNames, vidScores = readDataFiles.readScores('shotScores.csv')
    #Number of frames skipped between subsequent frames
    CF = 5
    for i in range(len(vidNames)):
        parsedVid, parsedScore = removeBadFrames(vidNames[i], vidScores[i],
                                    desired_w, desired_h)
        #parsedList.append(parseCF(parsedVid, parsedScore, CF))
        parsedVid, parsedScore = parseCF(parsedVid, parsedScore, CF)
        vggOut = np.zeros((len(parsedVid), 512, 7, 7))
        for count in range(len(parsedVid)):
            #print(parsedVid[i].shape)
            #print(torch.tensor(np.swapaxes(parsedVid[i], 0, 2)).unsqueeze(0).shape)
            vggOut[count,...] = vgg16_fcn(torch.tensor(np.swapaxes(parsedVid[count], 0, 2))
                                                .unsqueeze(0).float().cuda())

        torch.save([vggOut, parsedScore] ,'tensors/'+vidNames[i]+'.pt')
        del parsedVid
        del parsedScore
    #parsedList = torch.load('tensors/xxdtq8mxegs.pt')

    #for i in range(len(parsedList)):
        #print(len(parsedList[i][0]), len(parsedList[i][1]))
        #plt.imshow(parsedList[i][0][100])
        #plt.show()

    print("~~ Done ~~")


if __name__ == "__main__":
    main()
