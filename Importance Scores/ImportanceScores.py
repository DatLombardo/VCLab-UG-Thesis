"""
Michael Lombardo
Thesis
Importance Score
"""

def removeTesting(data, category):
    trainValid = []
    test = []
    for val in data:
        if val[1] == category:
            test.append(val)
        else:
            trainValid.append(val)
    return trainValid, test

def seperateVideos(data):
    vidNames = []
    dataContainer = [[],[], [], [], []]
    for val in data:
        if not(val[0] in vidNames):
            vidNames.append(val[0])
        dataContainer[vidNames.index(val[0])].append(val)
    return vidNames, dataContainer


def main(filename, path, testSize, batchSize):
    import torch
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import readDataFiles as readDataFiles
    import ast
    import random

    class ImportanceScore(Dataset):
        def __init__(self, data, transforms=None):
            #Example of item: ['xxdtq8mxegs', '0', '4', '[1.25 1.25 1.25 1.25]']
            self.data = data
            self.transforms = transforms

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            segment = self.data[idx]
            loadVid = torch.load(path + segment[0] + ".pt")
            frames = torch.tensor(loadVid[0][int(segment[2]):int(segment[3])], dtype=torch.float32)
            #scores = torch.tensor(loadVid[1][int(segment[1]):int(segment[2])], dtype=torch.float32)
            scores = torch.tensor(loadVid[1][int(segment[3])], dtype=torch.float32)
            del loadVid
            if self.transforms is not None:
                frames = self.transforms(frames)

            return {'video': frames, 'score': scores}

    print("\n~~~| ImportanceScore.py Execution |~~~")
    segmentData = readDataFiles.readVideoDataCSV(filename)
    categories = ["VT", "VU", "GA", "MS", "PK", "PR", "FM", "BK", "BT", "DS"]
    cat = categories[random.randint(0,len(categories)-1)]
    #Need to seperate one category of videos to be used purely for testing.
    dataTrV, dataTe = removeTesting(segmentData, cat)
    #Seperates each video so that each can be tested and recorded seperately.
    names, dataTe = seperateVideos(dataTe)

    trainCount = int(testSize * len(dataTrV))
    validCount = len(dataTrV) - trainCount
    trainData, validData = torch.utils.data.random_split(dataTrV, [trainCount, validCount])

    print("Loaded dataset")

    testLoader = []
    trainLoader = DataLoader(ImportanceScore(trainData), batch_size=batchSize, shuffle=True)
    validLoader = DataLoader(ImportanceScore(validData), batch_size=batchSize, shuffle=True)
    for video in dataTe:
        testLoader.append(DataLoader(ImportanceScore(video), batch_size=batchSize, shuffle=None))


    print("~~~| ImportanceScore.py Complete |~~~\n")
    return trainLoader, validLoader, testLoader, cat, names


if __name__ == "__main__":
    main()
