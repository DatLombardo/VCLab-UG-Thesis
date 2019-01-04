"""
Michael Lombardo
Thesis
Custom TVSum50 Dataset
ReadDataFiles
"""

def readSegmentCSV(filename):
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

def readVideoCSV(filename):
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
