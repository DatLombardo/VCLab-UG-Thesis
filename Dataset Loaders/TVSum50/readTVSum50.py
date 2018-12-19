"""
Michael Lombardo
HSA-RNN - Thesis
TVSum50 Dataset
"""
#newData, k(# frame per segments), nSegments, width, height
def main(newData, k, nSegments, doDistance , width, height):
    import os
    import torch
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, utils
    import time
    #Dataloading
    import readDataFiles as readDataFiles
    import videoToTensor as videoToTensor
    import processVideoData as processVideoData

    """
    TVSum50 Dataloader
    """
    class TVSumDataset(Dataset):
        def __init__(self, info, scores, segments, frameCount, transforms=None):
            """
            Args:
                info: Video filenames.
                scores: 1 x 50 x videoFrames.
                segments:1 x 50 x 20 x k - k: Number of frames per segment.
                frameCount: Number of frames in each segment.
            """
            self.info = info
            self.scores = scores
            self.segments = segments
            self.frameCount = frameCount
            self.transforms = transforms

        def __len__(self):
            return len(self.segments)

        def __getitem__(self, idx):
            filename, framenum, nframes = self.segments[idx]
            fileScoreIdx = self.info.index(filename)
            segScores = self.scores[fileScoreIdx][framenum:framenum+nframes]
            segFrames = torch.load("tensors/" + str(filename) + str(framenum) + ".pt")

            if self.transforms is not None:
                segFrames = self.transforms(segFrames)

            return {'video': segFrames, 'scores': segScores}

    """
    Usage within main()
    """
    if newData:
        processVideoData.main(k, nSegments, doDistance) #(k (Num frame per segments), nSegments, distance)
        videoToTensor.main(width, height) #(width, height)


    print("\n~~~| readTVSum50.py Execution |~~~")
    #vidInfo, scores = readDataFiles.readScoresCSV('shotScores.csv')
    vidInfo, scores = readDataFiles.readBoundaries('boundaries.csv')
    segments, fCount = readDataFiles.getSegments('videoData.csv')
    dataset = TVSumDataset(vidInfo, scores, segments, fCount)

    print("Loaded dataset")
    print("Number of segments: " + str(len(dataset)))



    debug = False
    if debug:
        print("vidInfo:", vidInfo[0])
        print("segments:", segments[0])

    """ Output of a sample segment

    #Testing Data
    item = dataset[3]
    itemSegment = item['video']
    for i in range(0,12,3):
        singleFrame = itemSegment[:,:,i:i+3]
        print(singleFrame.shape)
        imgplot = plt.imshow(singleFrame.numpy())
        plt.show(block=False)
        time.sleep(1)
        plt.close()

    """
    data_train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("~~~| readTVSum50.py Complete |~~~\n")
    #return dataset
    return data_train_loader


    """ Visualize each batch
    for batch_i, data in enumerate(data_train_loader):
        print('batch:', batch_i)
        print(len(data))
        for j in data.keys():
            print(j,len(data[j]))
        break
    """
