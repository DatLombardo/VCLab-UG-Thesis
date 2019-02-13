"""
Michael Lombardo
Thesis
Importance Scores Create New Data
"""
import ParseVideo as ParseVideo
import ParseSegments as ParseSegments

def main():

    newData = False
    """
    desired_w/h - Desired dimensions for VGG16 / Model
    CF - Number of frames skipped between subsequent frames
    newMean - Will create new mean.pt for processing the video, True if new.
    k - sequence length
    """
    width = 224
    height = 224
    CF = 5
    newMean = False #Needs to be True if using new data
    k = 4

    """
    Create video tensors of each video in the shotScores.csv
    Calculates the mean image, saves VGG16 output & scores of all frames
    Considering the cf value.
    """
    if newData:
        ParseVideo.main(width, height, CF, newMean)

    ParseSegments.main(k)
if __name__ == "__main__":
    main()
