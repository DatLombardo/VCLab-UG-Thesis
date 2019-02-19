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

k = 4
size = 400
tSize = 12

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
        xFlat = x.view((4,self.k,-1))

        #Output from LSTM 4 x 3 x 256
        lstmOut, _ = self.lstm(xFlat)
        #print(lstmOut.shape)
        lastOut = lstmOut[-1]

        sigOut = self.sigmoid(self.fc(lastOut))
        return sigOut

def accuracy(x,y):
    xy = 1 - (np.sum(abs(np.subtract(x,y)) > 0.05)/len(x))
    if xy <= 0:
        xy = 0
    return xy

ISLoader = ISDataloader.main('vidData/videoDataGA4.csv', 'tensors/', size) #for train dataloader
testLoader = ISDataloader.main('vidData/videoDataFull4.csv', 'tensors/', tSize) #for test dataloader



model = MyModel(512*7*7, 256, k).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss(reduction='sum').cuda()
with open('results/'+str(size)+'-'+str(k)+'.tsv', "w+") as tsv_file:
    writer = csv.writer(tsv_file, delimiter='\t')
    time = datetime.datetime.now()
    writer.writerow([str(time)])
    writer.writerow(["~~ Training Error ~~"])
    enums = 0
    num_epochs = 100
    for epoch in range(num_epochs):
        errors = []
        gtx = []
        gty = []
        genx = []
        geny = []
        for batch_i, batch_data in enumerate(ISLoader):
            x = Variable(torch.tensor(batch_data['video'], dtype=torch.float32)).cuda()
            y = Variable(torch.tensor(batch_data['score'], dtype=torch.float32)).cuda()
            optimizer.zero_grad()
            out = model(x)
            loss_fn = torch.nn.MSELoss(reduction='sum').cuda()
            error = loss_fn(out.flatten(), y).cuda()
            if epoch % 50 == 0:
                index = random.randint(0,3)
                errors.append(error.item())
                gty.append(y[index].detach().cpu().item())
                geny.append(out[index].detach().cpu().item())
                gtx.append(batch_i)
                genx.append(batch_i)
            enums += 1
            error.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print("epoch:" + str(epoch))
            writer.writerow(["Ground Truth: " + str(gty)])
            writer.writerow(["Generated: " + str(geny)])
            writer.writerow(["Epoch: " + str(epoch) + " Average Accuracy: "+  str(accuracy(gty, geny))])
            ax = plt.subplot(111)
            plt.scatter(gtx, gty, c='b', marker='x', label='gt')
            plt.scatter(genx, geny, c='r', marker='|', label='pred')
            ax.legend(loc='best')
            plt.xlabel("Item Number")
            plt.ylabel("Score")
            plt.ylim(0.0,1.0)
            plt.savefig("Images/"+str(size)+ "-epoch-"+ str(epoch)+"-train.jpg")
            plt.cla()

            #plt.show()


    #print("Number of Enumerations: ", enums)
    writer.writerow(["Number of Enumerations: " + str(enums)])
    #print("~~ Testing ~~")
    geny = []
    gty = []
    for batch_i, test_data in enumerate(testLoader):
        x = Variable(torch.tensor(test_data['video'], dtype=torch.float32)).cuda()
        y = Variable(torch.tensor(test_data['score'], dtype=torch.float32)).cuda()
        optimizer.zero_grad()
        out = model(x)
        loss_fn = torch.nn.MSELoss(reduction='sum').cuda()
        error = loss_fn(out.flatten(), y).cuda()
        #print("Actual: ", y.cpu().numpy())
        #print("Generated: ",out.detach().flatten().cpu().numpy())
        gty.append(y.cpu().numpy())
        geny.append(out.detach().flatten().cpu().numpy())
    ax = plt.subplot(111)
    gty = np.array(gty).flatten()
    geny = np.array(geny).flatten()
    #print(gty)
    #print(geny)
    #print("Accuracy: " + str(accuracy(gty, geny)))
    time = datetime.datetime.now()
    writer.writerow([str(time)])
    writer.writerow(["~~ Testing ~~"])
    writer.writerow(["Ground Truth: " + str(gty)])
    writer.writerow(["Generated: " + str(geny)])
    writer.writerow(["Average Accuracy: " +  str(accuracy(gty, geny))])
    plt.scatter(range(len(gty)), gty, c='b', marker='x', label='gt')
    plt.scatter(range(len(geny)), geny, c='r', marker='|', label='pred')
    ax.legend(loc='best')
    plt.xlabel("Item Number")
    plt.ylabel("Score")
    plt.ylim(0.0,1.0)
    plt.savefig("Images/"+str(size)+"-test.jpg")
    plt.cla()

    #plt.show()
