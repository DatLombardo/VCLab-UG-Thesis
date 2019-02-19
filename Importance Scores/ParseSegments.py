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
def main(k, filename):
    print("~~ Generataing Video Segments ~~")
    vidNames, vidCat, vidScores = readDataFiles.readCatScores(filename)
    segments = []
    for i in range(len(vidNames)):
        #Used to parse only GA category.
        #if vidCat[i] == 'GA':
            #[0] = frames, [1] = scores
            videoData = torch.load('tensors/'+vidNames[i]+'.pt')
            for frameNum in range(k,len(videoData[0])):
                segments.append([vidNames[i], frameNum-k, frameNum])

    print("~~ Writing Segment Names to videoData.csv ~~")
    with open('vidData/videoDataFull'+str(k)+'.csv', "w+") as csv_file:
        csv_file.truncate()
        writer = csv.writer(csv_file)
        for elem in segments:
            writer.writerow(elem)
    print("~~ Done ~~")




if __name__ == "__main__":
    main()
