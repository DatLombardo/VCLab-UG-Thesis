"""
Michael Lombardo
HSA-RNN - Thesis
useTVSumDataset
Sample python file for how to use TVSum Dataset when invoking for training / testing
"""

import readTVSum50 as TVSumData
import torchvision.models as models

#newData, k(# frame per segments), nSegments, Distance, width, height
#tvSumDataset = TVSumData.main(True, 30, 20, False, 224, 224) #for dataset
tvSumDataloader = TVSumData.main(False, 30, 20, False, 224, 224) #for dataloader
