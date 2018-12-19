import torch
import cv2
import numpy as np
def getSegments(filename):
      '''
      1-by-N(Segments)
      '''
      #Definition of empty container for segment / video csv storage
      segmentData = []
      videoData = []
      #Load csv file with pre-processed segments
      with open(filename) as infile:
          for line in infile:
              line = line.replace('\n','')
              data = line.split(",")
              # 0:filename, 1:startFrame, 2:frameCount
              videoData.append([data[0], int(data[1]), int(data[2])])
      #Set frameCount based on given length from pre-processing
      frameCount = videoData[0][2]
      return videoData, frameCount


def readScoresCSV(filename):
      '''
      1-by-N(videos)
      '''
      videoName = []
      #videoCategory = []
      videoScore = []
      with open('shotScores.csv') as infile:
          for line in infile:
              line = line.replace('"', '')
              line = line.replace('\n','')
              data = line.split(",")
              videoName.append(data[0])
              #videoCategory.append(data[1])
              videoScore.append(data[2:len(data)])
      return videoName, videoScore

def readBoundaries(filename):
      '''
      1-by-N(videos)
      '''
      videoName = []
      videoScore = []
      with open(filename) as infile:
          for line in infile:
              line = line.replace('"', '')
              line = line.replace('\n','')
              data = line.split(",")
              videoName.append(data[0])
              #videoCategory.append(data[1])
              boundaries = [int(i) for i in data[1:len(data)]]
              videoScore.append(boundaries)
      return videoName, videoScore

def getVideo(filename, startFrame, frameCount, width, height, mean):
    '''
    1-by-frameCount
    '''
    #Container for segment frames,
    #SegmentLength x width x height x 3(channels)
    #Double bracket to signify 1 parameter - dimensions
    segFrames = np.zeros((frameCount, width, height, 3))
    #Initialize opencv capture to read video frame-by-frame
    capture = cv2.VideoCapture("video/"+filename+".mp4")
    # Defining the starting point for reading.
    capture.set(1,startFrame);

    #Counter to break after 'frameCount' number of frames
    count = 0
    while (capture.isOpened()):
        ret, currFrame = capture.read()
        if ret:
            #Resize frame and convert to pytorch tensor.
            currFrame = cv2.resize(currFrame, (width, height))
            currFrame = np.divide(currFrame.astype(np.float32), 255.)
            currFrame = currFrame[...,[2,1,0]]
            currFrame = currFrame - mean
            m1 = np.min(currFrame)
            m2 = np.max(currFrame)
            d = m2-m1
            fixedFrame = (currFrame/d)-(m1)
            finalFrame = fixedFrame - np.min(fixedFrame)
            segFrames[count,...] = finalFrame
            count+=1
            if count == frameCount:
                break

    capture.release()
    return segFrames

def videoLength(filename):
    #Load mp4 file
    capture = cv2.VideoCapture("video/"+filename+".mp4")
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

""" Using Distance
def getVideo(filename, startFrame, frameCount, distance):
    #Container for segment frames
    tmp = torch.empty((240, 380, 3), dtype=torch.uint8)
    frameCollection = torch.tensor(tmp)

    #Initialize opencv capture to read video frame-by-frame
    capture = cv2.VideoCapture("video/"+filename+".mp4")
    capture.set(1,startFrame); # Defining the starting point for reading.
    #Counter to break after 'frameCount' number of frames
    count = frameCount
    while (capture.isOpened()):
        ret, currFrame = capture.read()
        if ret:
            #Resize frame and convert to pytorch tensor.
            f = torch.tensor(cv2.resize(currFrame, (380,240)))
            frameCollection = torch.cat((frameCollection, f), 2)
            count-=1
            if count == 0:
                break
        #Skip 9 frames to the next frame desired due to distance
        for i in range(distance-1):
            r, cF = capture.read()

    capture.release()
    return frameCollection[:,:,3:]
"""
