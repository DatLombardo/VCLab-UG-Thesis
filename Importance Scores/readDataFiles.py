"""
Michael Lombardo
Thesis
Importance Scores
ReadDataFiles
"""
import cv2
import numpy as np

def readVideoNamesCSV(filename):
      '''
      1-by-N(videos)
      '''
      videoName = []
      with open(filename) as infile:
          for line in infile:
              line = line.replace('"', '')
              line = line.replace('\n','')
              data = line.split(",")
              videoName.append(data[0])
      return videoName

def readVideoDataCSV(filename):
      '''
      1-by-N(videos)
      '''
      videoData = []
      with open(filename) as infile:
          for line in infile:
              line = line.replace('"', '')
              line = line.replace('\n','')
              data = line.split(",")
              videoData.append(data)
      return videoData

def readScores(filename):
      '''
      1-by-N(videos)
      '''
      videoData = []
      videoScores = []
      with open(filename) as infile:
          for line in infile:
              line = line.replace('"', '')
              line = line.replace('\n','')
              data = line.split(",")
              videoData.append(data[0])
              videoScores.append(data[2:])
      return videoData, videoScores

def darkLabel(frame):
    #Normalize to 0.0 - 1.0
    image = frame / 255.0

    #Determine Darkness coefficient Y = mean(RED * R + GREEN * G + BLUE * B)
    Y = ((image[:,:,0].mean() * 0.0722 ) +
                (image[:,:,1].mean() * 0.7152) +
                (image[:,:,2].mean() * 0.2126))

    #Compare Y value to empirically determined Sigma
    if (Y <= 0.097):
        return True

    return False

def blurryLabel(frame):
    #Horizontal Image Gradient
    gaussianX = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    #Vertical Image Gradient
    gaussianY = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

    #Compute S Coefficient - mean(Gx^2 + Gy^2)
    S = np.mean((gaussianX * gaussianX) + (gaussianY * gaussianY))

    #Compare S value to empirically determined Beta
    if (S <= 502.32):
        return True

    return False

def uniformLabel(frame):
    #128-bin histogram -> flatten into 1D array -> normalize
    #To get the max range of values use: np.ptp(gray_image)
    hist = (cv2.calcHist([frame],[0],None,[127],[0.0,255.0]).flatten())/ 255.087

    #Sort in descending order
    hist[::-1].sort()

    #Create ratio of each value with respect to hist
    ratios = np.divide(hist, np.sum(hist))

    #Compute U coefficient - sum of top 5th percentile
    U = 1 - np.sum(ratios[0:int(np.floor(0.05 * 128))])

    #Compare U value to empirically determined Y
    if (U < 0.2):
        True

    return False
