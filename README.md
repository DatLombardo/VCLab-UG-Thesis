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

2 &nbsp;&nbsp;&nbsp;&nbsp;Dependencies
============

* **numpy** : v1.15.3, Default matrix generation & array manipulation.
* **random** : v3.2, Generation of boundary values, segment start points.
* **torch** : v0.4.1, Processing tensors, saving & loading pytorch tensor files of segments.
* **csv** : v1.0, Reading & writing csv files.
* **cv2** : v3.4.3, Video Capture functions.

3 &nbsp;&nbsp;&nbsp;&nbsp;Video Processing
===============

3.1 &nbsp;&nbsp;&nbsp;&nbsp;Frame Manipulation
---------

Reading in frame as BGR; normalize, convert to RBG, and resize
```python
capture = cv2.VideoCapture("video.mp4")
while capture.isOpened():
    ret, cur_frame = capture.read()
    if ret:
        #Normalize values between 0.0-1.0
        cur_frame = np.divide(cur_frame.astype(np.float32), 255)
        #BGR to RGB
        cur_frame = cur_frame[...,[2,1,0]]
        #Resize
        resizedFrame = cv2.resize(cur_frame, (width, height))
```

Reading in frame similar to above, also considering mean image. Populating numpy array with required segment using a particular start point.
```python
segFrames = np.zeros((k, width, height, 3))

capture = cv2.VideoCapture("video.mp4")

capture.set(1, startPoint);
while capture.isOpened():
    ret, currFrame = capture.read()
    if ret:
        currFrame = cv2.resize(currFrame, (width, height))
        currFrame = np.divide(currFrame.astype(np.float32), 255.)
        currFrame = currFrame[...,[2,1,0]]
        #Subtract mean from frame
        currFrame = currFrame - mean
        #Normalize with respect to min/max of averaged frame
        m1 = np.min(currFrame)
        m2 = np.max(currFrame)
        d = m2-m1
        fixedFrame = (currFrame/d)-(m1)
        #Remove fixed frame's min again to normalize to 0.0 instead of new min.
        finalFrame = fixedFrame - np.min(fixedFrame)
        #Add current frame to previously populated array
        segFrames[totalFrames,...] = finalFrame
        totalFrames += 1
```

3.2 &nbsp;&nbsp;&nbsp;&nbsp;Mean Image
---------
The mean image of the dataset must be computed by adding all frames of the current dataset together and diving
```python
total, count = getMean(seg, desired_w, desired_h)
mean = np.divide(total, count)

#Within getMean()
#Read frame from dataset and resize frame accordingly.
curTotal = np.add(curTotal, resizedFrame)
```
4 &nbsp;&nbsp;&nbsp;&nbsp;PyTorch Save / Load Functions
============

4.1 &nbsp;&nbsp;&nbsp;&nbsp;Saving Pytorch Tensors
---------
Because creating segments for a pytorch dataset cannot be completed in real-time, and is storage expensive on memory the segments for the dataset must be written to disk and read when called. Memory issues arise with datasets that load segments on-the-fly without loading from a previously saved location.

A segment would be created similar to above in **3.1**. Next the segment would be saved with a detailed name so it can be easily found when the datasets __getitem__ is called.
```python
torch.save(mean, "mean.pt")

finalFrames = {'vid': segFrames, 'scores': [0,0,0,0,1,0,0,0,0]}
#Example filename used to show naming convention
#VidOneName[0:5] + startPointOne + _ + VidTwoName[0:5] + startPointTwo
torch.save(finalFrames, "3eYKf939LRw_o5954.pt")
segmentNames.append("3eYKf939LRw_o5954")
```

4.2 &nbsp;&nbsp;&nbsp;&nbsp;Loading Pytorch Tensors
---------
Loading of pytorch tensors would be used within loading in the mean image when processing video segments, and when the __getitem__ of the dataset is called to get an item.

```python
mean = torch.load("mean.pt")

class CustomDataset(Dataset):
      #init & __len__ removed

      def __getitem__(self, idx):
          segment = self.data[idx]
          #Load in tensor
          loadedPT = torch.load("tensors/" + str(segment[0]) + ".pt")
          #Convert to torch tensor for network
          frames = torch.tensor(loadedPT['vid'])
          scores = loadedPT['scores']
          if self.transforms is not None:
              frames = self.transforms(frames)
          return {'video': frames, 'scores': scores}
```
