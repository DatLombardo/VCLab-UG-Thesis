VCLab-Works
===============
Undergraduate Honours Thesis 2018-2019 in Visual Computing Lab

### Contributors  
* Michael Lombardo
* Faisal Qureshi

1 &nbsp;&nbsp;&nbsp;&nbsp;Introduction
============

This repository includes all work completed during my Undergraduate
Honours Thesis completed from September 2018 to April 2019 in the
Visual Computing Lab (VCLab).

The next step of my thesis will be exploring using LSTM's with the help of
VGG16 feature extraction to see if an LSTM can rank the importance of a given
frame after viewing a sequence of previous frames.

2 &nbsp;&nbsp;&nbsp;&nbsp;Dependencies
============

* **numpy** : v1.15.3, Default matrix generation & array manipulation.
* **random** : v3.2, Generation of boundary values, segment start points.
* **torch** : v0.4.1, Processing tensors, saving & loading pytorch tensor files of segments.
* **csv** : v1.0, Reading & writing csv files.
* **cv2** : v3.4.3, Video Capture functions.

3 &nbsp;&nbsp;&nbsp;&nbsp;Usage
============
To run the importance scores LSTM end-to-end with parsing data,
you must run the CreateData file to create the video data which
includes the frame sequences for each video. CreateData will also
create the data tensors for each video which can be used for ANY value of k
(sequence length). The data tensors will also be processed through VGG16 to
extract the feature vectors of each given frame, instead of saving the actual
RGB image saving lots of training time.

### Required Files / Folders
* **LSTM.py** : Contains the Model, Testing & Training
* **CreateData.py** : Creates data tensors, and video data for model usage.
* **ParseVideo.py** : Creates mean image, video tensors processed by VGG16
* **ParseSegments.py** : Creates videoData.csv's based on k value.
* **readDataFiles.py** : Csv reading dependency, worker file.
* **ImportanceScore.py** : Dataset & Dataloader


* **tensors/** : Contains data tensors
* **scores/** : Contains shot scores of dataset, and single category.
* **vidData/** : Contains csv files of each sequence of each video.
* **video/** : Contains raw .mp4 files of dataset.
* **results/** : Contains the written .tsv files of the training cycles.
* **Images/** : Contains .jpg images of the matplotlib figures generated.
* **weights/** : Contains saved weights from each run of the model.

If you do not wish to read the whole specification of the code, and already
have the data processed, simply run: **python3 LSTM.py** to execute the model.

3.1 &nbsp;&nbsp;&nbsp;&nbsp;Getting Video Data
---------
CreatData.py is used to create the video data, frame tensors, and the mean image
of the given dataset. If using TVSum50, the average shot scores of the 20
annotations can be found in the repository. I have also included shotScoresGA
which includes only ONE category.

### Parameters for creating data are all found in the file

```python
newData = False   #Will parse new tensors
width = 224
height = 224
CF = 5  #Cut frame, how many frames between each valid frame
newMean = False #Needs to be True if using new data
k = 4  #Sequence length
```
Based on the following parameters the two following files will execute

**Be sure to create a scores/ folder prior, the csv files will be written even
if they do not exist.**
```python
if newData:
      ParseVideo.main(width, height, CF, newMean)

ParseSegments.main(k, "scores/shotScoresFull.csv")
```

Execution is simply **python3 CreateData.py**, ensure newData & newMean are set
to True.

3.2 &nbsp;&nbsp;&nbsp;&nbsp;The Model
---------
The model can be found in LSTM.py. If you are curious of the usage, and want to
validate that all of the dependencies have been installed, please refer to the
Jupyter Notebook provided, which trains a small sub-sample of 12 batch items.
Please remember to use this model on a CUDA supported machine. A lot of
refactoring is required to be used on a CPU only machine (Simply remove all
.cuda() and treat cuda variables as normal numpy values).

### The Model
```python
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
      xFlat = x.view((4,self.k,-1))
      lstmOut, _ = self.lstm(xFlat)
      lastOut = lstmOut[-1]
      sigOut = self.sigmoid(self.fc(lastOut))
      return sigOut
```

3.3 &nbsp;&nbsp;&nbsp;&nbsp;Testing & Training
---------
Training and Testing is pretty straight forward with the given LSTM.py,
this file will do all of the heavy lifting including outputting images and
tsv writes at given epochs which need to be modified manually. If using SSH to
run this model on a server, simply use Vim.
```python
if epoch % 25 == 0:
    print("epoch:" + str(epoch))
```

Ensure that your file names of your processed data match what is contained in
the folder to load in your training and testing data, with the given size
parameters.

Currently at the stage of posting, I am only training the single category. Thus
the full dataset is a valid testing item. When passing the size & tSize, the
dataloader will automatically shuffle the data making the training and testing
segments random, ontop of being shuffled automatically by the dataloader. If
the whole dataset is to be used in the training, instead of passing an integer
value to your dataloader, pass "All" as shown below.

```python
#LSTM.py
k = 4
size = 400
tSize = 12

ISLoader = ISDataloader.main('vidData/videoDataGA4.csv', 'tensorsGA/', size) #for train dataloader
testLoader = ISDataloader.main('vidData/videoDataFull4.csv', 'tensors/', tSize) #for test dataloader

#ImportanceScore.py
if size == "All":
    dataset = ImportanceScore(segmentData)
  else:
    random.shuffle(segmentData)
    dataset = ImportanceScore(segmentData[0:size])
```

Don't be intimidated by the amount of code in the training / testing loops,
most of it is simply for data visualization and data recording.

### Testing / Training without any matplotlib / tsv writing..
```python

#~~ Model / Loss Definition ~~
model = MyModel(512*7*7, 256, k).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss(reduction='sum').cuda()

#~~ Testing ~~
num_epochs = 10
for epoch in range(num_epochs):
  #Iterate through the loaded data
  for batch_i, batch_data in enumerate(ISLoader):
    #Parse current item (x,y) to PyTorch Variable
    x = Variable(torch.tensor(batch_data['video'], dtype=torch.float32)).cuda()
    y = Variable(torch.tensor(batch_data['score'], dtype=torch.float32)).cuda()
    optimizer.zero_grad()
    #Pass x to Model
    out = model(x)
    loss_fn = torch.nn.MSELoss(reduction='sum').cuda()
    #Compute Loss with Ground Truth and Preditions
    error = loss_fn(out.flatten(), y).cuda()
    error.backward()
    optimizer.step()

#~~ Testing ~~
for batch_i, test_data in enumerate(testLoader):
  #Parse current item (x,y) to PyTorch Variable
  x = Variable(torch.tensor(test_data['video'], dtype=torch.float32)).cuda()
  y = Variable(torch.tensor(test_data['score'], dtype=torch.float32)).cuda()
  optimizer.zero_grad()
  #Pass x to Model
  out = model(x)
  loss_fn = torch.nn.MSELoss(reduction='sum').cuda()
  #Compute Loss with Ground Truth and Preditions
  error = loss_fn(out.flatten(), y).cuda()
  print("Actual: ", y.cpu().numpy())
  print("Generated: ",out.detach().flatten().cpu().numpy())

```

After you validate that all data is processed, and the correct folders exist
in the directory. Execution is simply **python3 CreateData.py**. I would
recommend validating that the model trains a small size such as 12 or 40, try
to use multiples of 4 (batch size)
