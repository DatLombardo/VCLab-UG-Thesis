"""
Michael Lombardo
Thesis
Custom TVSum50 Dataset
"""

def main(filename, path, size):
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
            frames = torch.tensor(loadVid[0][int(segment[1]):int(segment[2])], dtype=torch.float32)
            #scores = torch.tensor(loadVid[1][int(segment[1]):int(segment[2])], dtype=torch.float32)
            scores = torch.tensor(loadVid[1][int(segment[2])], dtype=torch.float32)
            del loadVid
            if self.transforms is not None:
                frames = self.transforms(frames)

            return {'video': frames, 'score': scores}

    print("\n~~~| ImportanceScore.py Execution |~~~")
    segmentData = readDataFiles.readVideoDataCSV(filename)
    if size == "All":
        dataset = ImportanceScore(segmentData)
    else:
        random.shuffle(segmentData)
        dataset = ImportanceScore(segmentData[0:size])
    print("Loaded dataset")

    data_train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("~~~| ImportanceScore.py Complete |~~~\n")
    #return dataset
    return data_train_loader


if __name__ == "__main__":
    main()
