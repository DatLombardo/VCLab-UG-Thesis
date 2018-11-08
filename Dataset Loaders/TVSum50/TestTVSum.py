"""
Michael Lombardo
HSA-RNN - Thesis
useTVSumDataset
Sample python file for how to use TVSum Dataset when invoking for training / testing
"""

import readTVSum50 as TVSumData
tvSumDataset = TVSumData.main()

testItem = tvSumDataset[4]
#print(testItem['video'].shape)
