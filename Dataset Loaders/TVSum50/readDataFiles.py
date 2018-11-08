import torch
import cv2
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

def getVideo(filename, startFrame, frameCount, width, height):
    '''
    1-by-frameCount
    '''
    #Container for segment frames
    tmp = torch.empty((height, width, 3), dtype=torch.uint8)
    frameCollection = torch.tensor(tmp)

    #Initialize opencv capture to read video frame-by-frame
    capture = cv2.VideoCapture("video/"+filename+".mp4")
    # Defining the starting point for reading.
    capture.set(1,startFrame);

    #Counter to break after 'frameCount' number of frames
    count = frameCount
    while (capture.isOpened()):
        ret, currFrame = capture.read()
        if ret:
            #Resize frame and convert to pytorch tensor.
            f = torch.tensor(cv2.resize(currFrame, (width, height)))
            frameCollection = torch.cat((frameCollection, f), 2)
            count-=1
            if count == 0:
                break

    capture.release()
    return frameCollection[:,:,3:]

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
