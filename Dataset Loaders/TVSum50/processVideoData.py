"""
Michael Lombardo
HSA-RNN - Thesis
TVSum50 Processing Video
Process Video Data
Create videoData.csv containing segments for training/testing
"""
def main(k, nSegments, useDistance):

    import cv2
    import csv
    import random
    #Dataloading
    import readDataFiles as myDataReader

    """ Using Distance
    def generateSegmentsDist(vidInfo, frameCount, numSegments, distance):
        segments = []
        for vid in vidInfo:
            capture = cv2.VideoCapture("video/"+vid+".mp4")
            length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(numSegments):
                start = random.randint(0,length-(frameCount*distance))
                segments.append([vid, start, frameCount])
        return segments
    """

    def generateSegments(vidInfo, frameCount, numSegments):
        segments = []
        #Iterate through each video in dataset
        for vid in vidInfo:
            capture = cv2.VideoCapture("video/"+vid+".mp4")
            length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            #Create a number of segments for each video
            for i in range(numSegments):
                start = random.randint(0,length - frameCount)
                segments.append([vid, start, frameCount])
        return segments

    """
    Usage within main()
    """
    #Number of frames per segments
    #k = 30
    #Segments per video
    #nSegments = 20
    print("\n~~~| processVideoData.py Execution |~~~")
    print("Reading.")
    if useDistance:
        #Distance of frames between eachother
        distance = 10
        #Using Distance
        #result = generateSegmentsDist(vidInfo, k, nSegments, distance)
    else:
        vidInfo, scores = myDataReader.readScoresCSV('shotScores.csv')
        result = generateSegments(vidInfo, k, nSegments)


    #Clear contents of videoData.csv
    fileClear = open("videoData.csv", "w")
    fileClear.truncate()
    fileClear.close()

    print("Writing.")
    with open('videoData.csv', "a") as csv_file:
        csv_file.truncate()
        writer = csv.writer(csv_file)
        for elem in result:
            writer.writerow(elem)

    print("~~~| processVideoData.py Complete |~~~\n")
