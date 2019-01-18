"""
Michael Lombardo
Thesis
Custom TVSum50 Dataset
"""

def main():
    import os
    import torch
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import readDataFiles as readDataFiles

    class CustomDataset(Dataset):
        def __init__(self, data, transforms=None):
            """
            Args:
                data: 1000 x 2 x k - k: Number of frames per segment.
            """
            self.data = data
            self.transforms = transforms

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            segment = self.data[idx]

            loadedPT = torch.load("tensors/" + str(segment[0]) + ".pt")
            frames = torch.tensor(loadedPT['vid'])
            scores = loadedPT['scores']
            if self.transforms is not None:
                frames = self.transforms(frames)

            return {'video': frames, 'scores': scores}

    print("\n~~~| Custom.py Execution |~~~")
    segmentData = readDataFiles.readSegmentCSV('segments.csv')
    dataset = CustomDataset(segmentData)
    print("Loaded dataset")

    data_train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("~~~| Custom.py Complete |~~~\n")
    #return dataset
    return data_train_loader
