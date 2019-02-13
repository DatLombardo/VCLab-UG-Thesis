"""
Michael Lombardo
Thesis
ParseSegments
"""
import readDataFiles as readDataFiles
import torch
import numpy as np
import csv

#cutFrame, sequence length
def main(k):
    print("~~ Generataing Video Segments ~~")
    vidNames, vidScores = readDataFiles.readScores('shotScores.csv')
    segments = []
    for i in range(len(vidNames)):
        #[0] = frames, [1] = scores
        videoData = torch.load('tensors/'+vidNames[i]+'.pt')
        for frameNum in range(k,len(videoData[0])):
            #(videoData[0][frameNum-k:frameNum])
            scores = videoData[1][frameNum-k:frameNum]
            segments.append([vidNames[i], frameNum-k, frameNum, scores])

    print("~~ Writing Segment Names to videoData.csv ~~")
    with open('videoData.csv', "a") as csv_file:
        csv_file.truncate()
        writer = csv.writer(csv_file)
        for elem in segments:
            writer.writerow(elem)
    print("~~ Done ~~")




if __name__ == "__main__":
    main()
