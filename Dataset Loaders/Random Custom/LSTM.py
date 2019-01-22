"""
Michael Lombardo
Thesis
Custom Dataset Worker

Execution is just simply running the program, generates a new set of data.
Presets are: k:9, nSegments:1000, width:224, height:224.
"""

import torch
import numpy as np
import torchvision.models as models
from torch.nn.parameter import Parameter
import Custom as CustomDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import csv
import datetime


def getDataPoint(element, index):
    scores = []
    for frame in range(len(element['video'][index])):
        video = element['video'][index]
        scores.append(int(element['scores'][frame][index]))
    points = np.zeros((7,3), dtype=int)
    frames = np.add(*np.indices((7, 3)))
    if (1 in scores):
        boundary = scores.index(1)
        for i in range(7):
            for j in range(3):
                frames[i][j] = i + j
                if (i + j == boundary):
                    points[i][j] = 1
    return points.tolist(), frames.tolist(), video, index

def parseVideoMatrix(vid, positions):
    vid = vid.numpy()
    newData = np.zeros((7, 3, 3, 224,224))
    count = 0
    for i in positions:
        frames = np.zeros((3, 3, 224, 224))
        for j in range(len(i)):
            #Models expect 3xHxW
            #Current format before swaps, WxHx3
            frames[j,...] = np.swapaxes(vid[i[j]], 0, 2)
        newData[count,...] = frames
        count += 1
    return newData

def parseViewVids(vid, positions):
    vid = vid.numpy()
    newData = np.zeros((7, 3, 224, 224,3))
    count = 0
    for i in positions:
        frames = np.zeros((3, 224, 224, 3))
        for j in range(len(i)):
            #Models expect 3xHxW
            #Current format before swaps, WxHx3
            frames[j,...] = vid[i[j]]
        newData[count,...] = frames
        count += 1
    return newData

def viewVideo(vid):
    for i in vid:
        for j in i:
            plt.imshow(j)
            plt.show(block=False)

def accuracy(x,y):
    try:
        np.testing.assert_array_equal(x, y)
        res = True
    except AssertionError as err:
        res = False
        print (err)


"""
USING SIGMOID
"""

class MyModel(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(MyModel, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        #vgg16 = models.vgg16(pretrained=True).cuda()
        for param in vgg16.parameters():
            param.requires_grad = False
            # Replace the last fully-connected layer
            # Parameters of newly constructed modules have requires_grad=True by default
        self.vgg16_fcn = vgg16.features
        #self.vgg16_fcn.cuda()
        self.lstm = torch.nn.LSTM(inputDim, outputDim, 1, True, True, 0.5);
        self.fc = nn.Linear(outputDim, 1)
        self.flatten_parameters()
        self.sigmoid = nn.Sigmoid()
        self.inputDim = inputDim

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def forward(self, x):
        newBatch = []
        for t in range(len(x[1])):
            newBatch.append(self.vgg16_fcn(x[:,t,:,:,:].float()))
        grad = False

        #4 x 3 x (512 x 7 x 7)
        vggOut = torch.stack(newBatch, 1).detach_()
        #print(vggOut.shape)

        #4 x 3 x 25088
        test = vggOut.view((4,3,-1))
        #print(test.shape)

        #Output from LSTM 4 x 3 x 256
        lstmOut, _ = self.lstm(test)
        #print(lstmOut.shape)

        #Output from Fully Connected Layer 4 x 3 x 1
        #fcOut = self.fc(lstmOut)
        #print(fcOut)
        #return fcOut

        sigOut = self.sigmoid(self.fc(lstmOut))
        #print(sigOut)
        return sigOut


"""
Do not run this code block without cuda()

Load in vgg16,
"""
vgg16 = models.vgg16(pretrained=True).cuda()
#vgg16 = models.vgg16(pretrained=True)
customDataloader = CustomDataset.main('segments.csv') #for test dataloader
testDataloader = CustomDataset.main('testSegments.csv') #for test dataloader


for param in vgg16.parameters():
    param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
vgg16_fcn = vgg16.features
vgg16_fcn.cuda()

fileClear = open("results.csv", "w")
fileClear.truncate()
fileClear.close()


"""
Using Cuda

"""

model = MyModel(512*7*7, 256).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

with open('results.csv', "w") as csv_file:
    csv_file.truncate()
    writer = csv.writer(csv_file)
    time = datetime.datetime.now()
    writer.writerow(str(time))

    num_epochs = 500
    for epoch in range(num_epochs):
        if (epoch % 10 == 0):
            print('epoch: ' + str(epoch))
        for batch_i, batch_data in enumerate(customDataloader):
            print('batch: ' + str(batch_i))
            for i in range(len(batch_data['video'])):
                scoreList, frameNums, vidData, index = getDataPoint(batch_data, i)
                dataItem = parseVideoMatrix(vidData, frameNums)
                data = [dataItem[0:4], dataItem[3:]]
                #Ground Truth
                GT = [np.squeeze(scoreList[0:4]), np.squeeze(scoreList[3:])]
                for j in range(2):
                    x = Variable(torch.tensor(data[j], dtype=torch.float32)).cuda()
                    y = Variable(torch.tensor(GT[j], dtype=torch.float32)).cuda()
                    optimizer.zero_grad()
                    out = model(x)
                    out = out.squeeze()
                    error = nn.functional.binary_cross_entropy(input=out, target=y, reduce=True).cuda()
                    error.backward()
                    optimizer.step()
                    if ((epoch) % 100 == 0) and (batch_i % 25 == 0):
                        writer.writerow('item: ' + str(i+1) + ' epoch:' + str(epoch) + '\n\tbatch: ' + str(batch_i) + ', error: ' + str(error.item()))

    testData = []
    for batch_i, data in enumerate(testDataloader):
        for i in range(len(data['video'])):
            scoreList, frameNums, vidData, index = getDataPoint(data, i)
            dataItem = parseVideoMatrix(vidData, frameNums)
            testData.append([scoreList, dataItem])
            print(dataItem.shape)
            print(scoreList)
        if batch_i == 1:
            break

    for i in range(len(testData)):
        #Within batchData - batchData[0]: score, batchData[1]:video data
        data = [testData[i][1][0:4], testData[i][1][3:]]
        #Ground Truth
        GT = [np.squeeze(testData[i][0][0:4]), np.squeeze(testData[i][0][3:])]
        for j in range(2):
            x = Variable(torch.tensor(data[j], dtype=torch.float32)).cuda()
            y = Variable(torch.tensor(GT[j], dtype=torch.float32)).cuda()
            optimizer.zero_grad()
            out = model(x)
            out = out.squeeze()
            error = nn.functional.binary_cross_entropy(input=out, target=y, reduce=True).cuda()
            error.backward()
            optimizer.step()
            print('item: ' + str(i+1) + '\n\tbatch: ' + str(j) + ', error: ' + str(error.item()))
            writer.writerow("---Results--")
            writer.writerow('item: ' + str(i+1) + '\n\tbatch: ' + str(j) + ', error: ' + str(error.item()))
    time = datetime.datetime.now()
    writer.writerow(str(time))
