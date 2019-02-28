"""
Michael Lombardo
Thesis
Importance Score
"""

import torch
import numpy as np
import torchvision.models as models
import ImportanceScores as ISDataloader
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import csv
import datetime
import random

k = 16
trainRatio = 0.8
batchSize = 16
threshold = 0.05

class MyModel(nn.Module):
    def __init__(self, inputDim, outputDim, k):
        super(MyModel, self).__init__()
        self.lstm = torch.nn.LSTM(inputDim, outputDim, 1, True, True, 0.5);
        self.fc = nn.Linear(outputDim, 1)
        self.flatten_parameters()
        self.sigmoid = nn.Sigmoid()
        self.inputDim = inputDim
        self.k = k

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def forward(self, x):
        #x = 4 x k x (512 x 7 x 7) <- Needs to be flattened

        #4 x k x 25088
        xFlat = x.view((self.k,len(x),-1))
        #Output from LSTM 4 x 3 x 256
        lstmOut, _ = self.lstm(xFlat)
        lastOut = lstmOut[-1]

        sigOut = self.sigmoid(self.fc(lastOut))
        return sigOut

def accuracy(x,y):
    #xy = 1 - (sum(abs(subtract(x,y)) > threshold)/len(x))
    sum = torch.tensor(torch.sum(torch.abs(torch.sub(torch.tensor(x),torch.tensor(y))) > threshold), dtype=torch.float32)
    diff = torch.div(sum,torch.tensor(len(x), dtype=torch.float32))
    xy = torch.tensor(torch.sub(torch.tensor(1,dtype=torch.float32), diff), dtype=torch.float32)
    if xy <= 0:
        xy = 0
    return xy

ISLoader, ValidLoader, TestLoader, category, tNames = ISDataloader.main('vidData/videoData16.csv', 'tensors/', trainRatio, batchSize)

model = MyModel(512*7*7, 256, k).cuda()
model.train()

#Load previous weights to continue training.
#model.load_state_dict(torch.load('weights.pt'))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss(reduction='sum').cuda()

with open('results/b-'+str(batchSize)+ '-k-'+ str(k)+'-t-'+str(threshold)+'.tsv', 'w+') as tsv_file:
    writer = csv.writer(tsv_file, delimiter='\t')
    time = datetime.datetime.now()
    writer.writerow(['Batch Size: ' + str(batchSize), 'Seq Len: ' + str(k)])
    writer.writerow(['Testing Category: ' + category, 'Threshold: ' + str(threshold)])
    writer.writerow(['Training Size: ' + str(len(ISLoader.dataset)),
                        'Validation Size: ' + str(len(ValidLoader.dataset))])
    writer.writerow([str(time)])
    writer.writerow(['~~ Training / Validation ~~'])
    enums = 0
    num_epochs = 50
    for epoch in range(num_epochs):
        gty = []
        geny = []
        count = 0
        avgLoss = 0
        #Training Loop
        for batch_i, batch_data in enumerate(ISLoader):
            x = Variable(torch.tensor(batch_data['video'], dtype=torch.float32)).cuda()
            y = Variable(torch.tensor(batch_data['score'], dtype=torch.float32)).cuda()
            optimizer.zero_grad()
            out = model(x)
            error = loss_fn(out.flatten(), y)
            error.backward()
            optimizer.step()
            if (epoch % 10 == 0) or (epoch == 49):
                # Collect loss
                avgLoss += error.item()
                count += 1
                gty.extend(y.detach())
                geny.extend(out.detach())
            enums += 1

        if (epoch % 10 == 0) or (epoch == 49):
            #Recording Training Results
            print('epoch:' + str(epoch))
            avgLoss = avgLoss / count
            writer.writerow(['Epoch: ' + str(epoch) + ' Average Training Accuracy: '+  str(accuracy(gty, geny))])
            writer.writerow(['Average Training Loss: ' + str(avgLoss)])

            #Recording Validation Results
            validLoss = 0
            validCount = 0
            validgty = []
            validgeny = []
            model.eval()
            #Validation Loop
            for batch_i, valid_data in enumerate(ValidLoader):
                x = Variable(torch.tensor(valid_data['video'], dtype=torch.float32)).cuda()
                y = Variable(torch.tensor(valid_data['score'], dtype=torch.float32)).cuda()

                out = model(x)
                error = loss_fn(out.flatten(), y)

                validgty.extend(y)
                validgeny.extend(out.detach().flatten())
                validLoss += error.item()
                validCount += 1
            validLoss = validLoss / validCount
            writer.writerow(['Epoch: ' + str(epoch) + ' Average Validation Accuracy: '+  str(accuracy(validgty, validgeny))])
            writer.writerow(['Average Validation Loss: ' + str(validLoss)])
            model.train()

    writer.writerow(['Number of Testing Enumerations: ' + str(enums)])
    time = datetime.datetime.now()
    writer.writerow([str(time)])
    writer.writerow(['~~ Testing ~~'])
    torch.save(model.state_dict(), 'weights/b-'+str(batchSize)+'-k-'+ str(k)+'-t-'+str(threshold)+'.pt')
    totalAvg = 0
    model.eval()
    #Testing Loop
    for i in range(len(TestLoader)):
        geny = []
        gty = []
        for batch_i, test_data in enumerate(TestLoader[i]):
            x = Variable(torch.tensor(test_data['video'], dtype=torch.float32)).cuda()
            y = Variable(torch.tensor(test_data['score'], dtype=torch.float32)).cuda()

            out = model(x)
            error = loss_fn(out.flatten(), y)

            gty.extend(y.detach())
            geny.extend(out.detach().flatten())
        gty = np.array(gty).flatten()
        geny = np.array(geny).flatten()

        writer.writerow(['Video: ' + tNames[i], "Size: " + str(len(TestLoader[i].dataset))])
        writer.writerow(['Ground Truth: ' + str(gty)])
        writer.writerow(['Generated: ' + str(geny)])
        totalAvg += accuracy(gty, geny)
        writer.writerow(['Accuracy: ' +  str(accuracy(gty, geny))])
    writer.writerow(['Average Accuracy: ' +  str(totalAvg / len(TestLoader))])
