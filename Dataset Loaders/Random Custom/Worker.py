"""
Michael Lombardo
Thesis
Custom Dataset Worker

Execution is just simply running the program, generates a new set of data.
Presets are: k:9, nSegments:1000, width:224, height:224.
"""
import random
import cv2
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
import time
import readDataFiles as readDataFiles

#Generate the segment if it will have a boundary, get the videos and start points
def generateSegments(nSegments, videoList):
    segments = []
    #Iterate through each video in dataset
    for i in range(nSegments):
        boundary = random.randint(0,1)
        if boundary == 1:

            vidOne = videoList[random.randint(0, len(videoList)-1)]
            vidTwo = videoList[random.randint(0, len(videoList)-1)]
            capOne = cv2.VideoCapture("video/"+vidOne+".mp4")
            startOne = random.randint(0, int(capOne.get(cv2.CAP_PROP_FRAME_COUNT))- 60)
            capOne.release()

            capTwo = cv2.VideoCapture("video/"+vidTwo+".mp4")
            startTwo = random.randint(0, int(capTwo.get(cv2.CAP_PROP_FRAME_COUNT))- 60)
            capTwo.release()

            boundary = random.randint(1, 8)
            scores = np.zeros(9, dtype=int)
            scores[boundary] = 1

            segments.append([vidOne, startOne, vidTwo, startTwo, scores.tolist()])
        else:
            vidOne = videoList[random.randint(0, len(videoList)-1)]
            capOne = cv2.VideoCapture("video/"+vidOne+".mp4")
            startOne = random.randint(0, int(capOne.get(cv2.CAP_PROP_FRAME_COUNT))- 60)
            scores = np.zeros(9, dtype=int)
            segments.append([vidOne, startOne, "", 0, scores.tolist()])
            capOne.release()

    return segments



def getMean(videoData, width, height):
    curTotal = np.zeros((width, height, 3))
    curCount = 0
    for elem in videoData:
        if (curCount % 50 == 0):
            print(str(curCount/len(videoData)*10) + "%")
        if (len(elem[2]) == 0):
            capOne = cv2.VideoCapture("video/"+elem[0]+".mp4")
            startOne = elem[1]
            count = 0
            capOne.set(1, startOne);
            while capOne.isOpened():
                ret, cur_frame = capOne.read()
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
            capOne.release()

        else:
            capOne = cv2.VideoCapture("video/"+elem[0]+".mp4")
            startOne = elem[1]
            startTwo = elem[3]
            boundPoint = elem[4].index(1)
            count = 0
            capOne.set(1, startOne);
            while capOne.isOpened():
                ret, cur_frame = capOne.read()
                if ret:
                    #Normalize values between 0.0-1.0
                    cur_frame = np.divide(cur_frame.astype(np.float32), 255)

                    #List manipulation, taking all previous index's as first parameter
                    #using ..., then manipulating frames from BGR to RGB.
                    cur_frame = cur_frame[...,[2,1,0]]

                    #Covert frame to expected dimensions, consider loss of quality.
                    resizedFrame = cv2.resize(cur_frame, (width, height))
                    curCount+=1
                    if count == boundPoint:
                        break
                    count+= 1
                    #print(type(frame), frame.shape)
                    curTotal = np.add(curTotal, resizedFrame)
                else:
                    break
            capOne.release()
            capTwo = cv2.VideoCapture("video/"+elem[2]+".mp4")
            capTwo.set(1, startTwo);
            while capTwo.isOpened():
                ret, cur_frame = capTwo.read()
                if ret:
                    #Normalize values between 0.0-1.0
                    cur_frame = np.divide(cur_frame.astype(np.float32), 255)

                    #List manipulation, taking all previous index's as first parameter
                    #using ..., then manipulating frames from BGR to RGB.
                    cur_frame = cur_frame[...,[2,1,0]]

                    #Covert frame to expected dimensions, consider loss of quality.
                    resizedFrame = cv2.resize(cur_frame, (width, height))
                    curCount+=1
                    if count == 8:
                        break
                    count+=1
                    #print(type(frame), frame.shape)
                    curTotal = np.add(curTotal, resizedFrame)
                else:
                    break
            capTwo.release()

    return curTotal, curCount

#Takes determined boundary locations (videoData) & mean to determine segments
#Writes result to tensors
def createSegments(k, videoData, width, height, mean):
    segmentNames = []
    updateCount = 0
    #SegmentLength x width x height x 3(channels)
    for elem in videoData:
        totalFrames = 0
        if (updateCount % 50 == 0):
            print(str(updateCount/len(videoData)*100) + "%")
        #SegmentLength x width x height x 3(channels)
        #Double bracket to signify 1 parameter - dimensions

        segFrames = np.zeros((k, width, height, 3))

        if (len(elem[2]) == 0):
            capOne = cv2.VideoCapture("video/"+elem[0]+".mp4")
            startOne = elem[1]

            capOne.set(1, startOne);
            while capOne.isOpened():
                ret, currFrame = capOne.read()
                if ret:
                    currFrame = cv2.resize(currFrame, (width, height))
                    currFrame = np.divide(currFrame.astype(np.float32), 255.)
                    currFrame = currFrame[...,[2,1,0]]
                    currFrame = currFrame - mean
                    m1 = np.min(currFrame)
                    m2 = np.max(currFrame)
                    d = m2-m1
                    fixedFrame = (currFrame/d)-(m1)
                    finalFrame = fixedFrame - np.min(fixedFrame)

                    segFrames[totalFrames,...] = finalFrame
                    #segFrames[totalFrames,:,:,:] = finalFrame
                    totalFrames+=1
                    if totalFrames == k:
                        break
                else:
                    break
            capOne.release()
            finalFrames = {'vid': segFrames, 'scores': elem[4]}
            torch.save(finalFrames, "tensors/" + str(elem[0][0:5]) + str(elem[1]) + ".pt")
            segmentNames.append([str(elem[0][0:5]) + str(elem[1]), elem[4]])

        else:
            capOne = cv2.VideoCapture("video/"+elem[0]+".mp4")
            startOne = elem[1]
            startTwo = elem[3]
            boundPoint = elem[4].index(1)

            capOne.set(1, startOne);
            while capOne.isOpened():
                ret, currFrame = capOne.read()
                if ret:
                    currFrame = cv2.resize(currFrame, (width, height))
                    currFrame = np.divide(currFrame.astype(np.float32), 255.)
                    currFrame = currFrame[...,[2,1,0]]
                    currFrame = currFrame - mean
                    m1 = np.min(currFrame)
                    m2 = np.max(currFrame)
                    d = m2-m1
                    fixedFrame = (currFrame/d)-(m1)
                    finalFrame = fixedFrame - np.min(fixedFrame)
                    segFrames[totalFrames,...] = finalFrame
                    #segFrames[totalFrames,:,:,:] = finalFrame
                    #segFrames[totalFrames] = finalFrame
                    if totalFrames == boundPoint:
                        break
                    totalFrames += 1

                else:
                    break
            capOne.release()

            capTwo = cv2.VideoCapture("video/"+elem[2]+".mp4")
            capTwo.set(1, startTwo);
            while capTwo.isOpened():
                ret, currFrame = capTwo.read()
                if ret:
                    currFrame = cv2.resize(currFrame, (width, height))
                    currFrame = np.divide(currFrame.astype(np.float32), 255.)
                    currFrame = currFrame[...,[2,1,0]]
                    currFrame = currFrame - mean
                    m1 = np.min(currFrame)
                    m2 = np.max(currFrame)
                    d = m2-m1
                    fixedFrame = (currFrame/d)-(m1)
                    finalFrame = fixedFrame - np.min(fixedFrame)
                    segFrames[totalFrames,...] = finalFrame
                    #segFrames[totalFrames,:,:,:] = finalFrame
                    #segFrames[totalFrames] = finalFrame
                    if totalFrames == k-1:
                        break
                    totalFrames += 1

                else:
                    break
            capTwo.release()
            finalFrames = {'vid': segFrames, 'scores': elem[4]}
            torch.save(finalFrames, "tensors/" + str(elem[0][0:5]) + str(elem[1])
                                    + str(elem[2][0:5]) + str(elem[3]) + ".pt")
            segmentNames.append([str(elem[0][0:5]) + str(elem[1])
                                + str(elem[2][0:5]) + str(elem[3]), elem[4]])

        updateCount += 1
    return segmentNames



def main():
    desired_w, desired_h = 224, 224

    #Reads in list of potential videos to make custom segments with.
    segmentData = readDataFiles.readVideoNamesCSV('shotScores.csv')

    print("~~ Generating Segments ~~")
    #Generate segments for boundary detection.
    seg = generateSegments(1000, segmentData)

    #Clear contents of videoData.csv
    fileClear = open("videoData.csv", "w")
    fileClear.truncate()
    fileClear.close()

    print("~~ Writing Segments to videoData.csv ~~")
    with open('videoData.csv', "a") as csv_file:
        csv_file.truncate()
        writer = csv.writer(csv_file)
        for elem in seg:
            writer.writerow(elem)
    print("~~ Done ~~")

    #Generate Mean
    print("~~ Generating Mean ~~")

    total, count = getMean(seg, desired_w, desired_h)
    mean = np.divide(total, count)
    #plt.imshow(mean)
    #plt.show(block=False)
    meanTensor = torch.tensor(mean)
    torch.save(mean, "mean.pt")
    print("~~ Done ~~")

    print("~~ Creating Segments ~~")
    segNames = createSegments(9, seg, desired_w, desired_h, mean)
    print("~~ Done ~~")
    fileClear = open("segments.csv", "w")
    fileClear.truncate()
    fileClear.close()

    print("~~ Writing Segment Names to segments.csv ~~")
    with open('segments.csv', "a") as csv_file:
        csv_file.truncate()
        writer = csv.writer(csv_file)
        for elem in segNames:
            writer.writerow(elem)
    print("~~ Done ~~")


if __name__ == "__main__":
    main()
