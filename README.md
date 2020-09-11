# PoseClassifer

## Intro
This repo is a simple neural network classifier that classified 13 human pose classes.<br/>

Given 14 body joints as `node:[x1,y1,z1,v],left_eye:[x2,y2,z2,v],....,right_ankle:[x14,y14,z14,v]` similar to [COCO data format](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format), the model will output one a pose like `stand_up` or`sit-down` on each testing data.  
<br/>
(v: visibility, if the joint is visible, v=1, otherwise, v=0 )<br/>
 
## Feature
1. Pytorch <br/>
2. Python 3.6 <br/>

## Network Structure
![](network.png "network outline")

## Sample result
![](dataFormat.png "data format")
![](sample.png "keypoints visualized in 3D space")

## Training result
![](training_result/all_loss_128_20000.jpg "training,validation,testing loss")
![](training_result/acc_128_20000.jpg "Accuracy")
