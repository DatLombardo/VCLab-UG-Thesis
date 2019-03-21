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

Presets are: cf: 5, width:224, height:224 which are used from CreateData.py
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
    curTotal = np.zeros((width, height, 3))
    curCount = 0
    for i in range(len(videoData)):
        print(str(i/len(videoData)*100) + "%")

        #Grab current video .mp4 file for each frame
        cap = cv2.VideoCapture("video/"+videoData[i]+".mp4")

        while cap.isOpened():
            ret, currFrame = cap.read()
            if ret:
                #Normalize values between 0.0-1.0
                currFrame = np.divide(currFrame.astype(np.float32), 255)

                #List manipulation, taking all previous index's as first parameter
                #using ..., then manipulating frames from BGR to RGB.
                currFrame = currFrame[...,[2,1,0]]

                #Covert frame to expected dimensions, consider loss of quality.
                resizedFrame = cv2.resize(currFrame, (width, height))
                curCount += 1
                #print(type(frame), frame.shape)
                curTotal = np.add(curTotal, resizedFrame)
            else:
                break
        cap.release()

    return curTotal, curCount


def removeBadFrames(video, scores, w, h, mean, cf):
    #Grab current video .mp4 file for each frame
    cap = cv2.VideoCapture("video/"+video+".mp4")
    print("~~~ Parsing " + video + " ~~~")

    count = 0
    includeFrame = 1
    newVid = []
    newScores = []
    while cap.isOpened():
        ret, currFrame = cap.read()
        if count == 3001:
            break
        if ret:
            if (includeFrame == cf):
                currFrame = cv2.resize(currFrame, (w, h))[...,[2,1,0]]
                if readDataFiles.darkLabel(currFrame):
                    #print(count, 'Dark')
                    includeFrame -= 1
                elif readDataFiles.blurryLabel(currFrame):
                    #print(count, 'Blurry')
                    includeFrame -= 1
                elif readDataFiles.uniformLabel(currFrame):
                    #print(count, 'Uniform')
                    includeFrame -= 1
                else:
                    currFrame = np.divide(currFrame.astype(np.float32), 255.)
                    #Take mean from video, fix min / max values
                    currFrame = currFrame - mean
                    m1 = np.min(currFrame)
                    m2 = np.max(currFrame)
                    d = m2-m1
                    fixedFrame = (currFrame/d)-(m1)
                    finalFrame = fixedFrame - np.min(fixedFrame)
                    """ Frame Validation
                    if count % 100 == 0:
                        plt.imshow(finalFrame)
                        plt.show()
                    """
                    if (len(newVid) % 200 == 0):
                        print(len(newVid), len(newScores))

                    #Add valid frame to parsed list
                    newScores.append(scores[count])
                    newVid.append(currFrame)

                    #Reset CF counter
                    includeFrame = 0
            count += 1
            includeFrame += 1
        else:
            break

    cap.release()
    print("Final Length of: ", video, len(newVid), len(newScores))
    return newVid, np.array(newScores, dtype=float)


def main(desired_w, desired_h, CF, newMean):
    """
    Predefined Parameters
    #desired_w/h - Desired dimensions for VGG16 / Model
    #CF - Number of frames skipped between subsequent frames
    #newMean - Will create new mean.pt for processing the video, True if new.
    """

    print("~~ Generating Video Tensors ~~")

    vgg16 = models.vgg16(pretrained=True).cuda()
    for param in vgg16.parameters():
        param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
    vgg16_fcn = vgg16.features.cuda()
    vgg16_fcn.cuda()

    #Reads in list of potential videos to make custom segments with.
    vidNames, vidScores = readDataFiles.readScoresCSV('scores/OneAnnot.csv')
    print("~~ Generating Mean Image ~~")
    if newMean:
        total, count = getMean(vidNames, desired_w, desired_h)
        mean = np.divide(total, count)
        torch.save(mean, "mean1.pt")
        #plt.imshow(mean)
        #plt.show()
    else:
        mean = torch.load("mean1.pt")

    for i in range(len(vidNames)):
        parsedVid, parsedScore = removeBadFrames(vidNames[i], vidScores[i],
                                    desired_w, desired_h, mean, CF)
        #Divide by 5 to normalize values between 0 and 1 instead of 1
        normalizedScore = np.divide(parsedScore, 5.0)

        print("~~~ Extracting Features for " + vidNames[i] + " ~~~")
        #parsedList.append(parseCF(parsedVid, parsedScore, CF))
        #parsedVid, parsedScore = parseCF(parsedVid, parsedScore, CF)
        vggOut = np.zeros((len(parsedVid), 512, 7, 7))
        for count in range(len(parsedVid)):
            #print(parsedVid[i].shape)
            #print(torch.tensor(np.swapaxes(parsedVid[i], 0, 2)).unsqueeze(0).shape)
            if count % 200 == 0:
                print(vidNames[i] + " : " + str(count))
            vggOut[count,...] = vgg16_fcn(torch.tensor(np.swapaxes(parsedVid[count], 0, 2))
                                                .unsqueeze(0).float().cuda())
        print("Saving..")
        #Save VGG16 output tensors with ground truth for training
        torch.save([vggOut, normalizedScore] ,'tensors1/'+vidNames[i]+'.pt')
        print("Done")
        #Uncomment if you wish to view the resulting video frames
        #torch.save([parsedVid, parsedScore], 'tensors/'+vidNames[i]+'Vid.pt')
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
