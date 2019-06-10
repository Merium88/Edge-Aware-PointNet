# Edge-Aware-PointNet
### Introduction
This work is based on our paper (https://www.researchgate.net/publication/333676783_EPN_Edge-Aware_PointNet_for_Object_Recognition_from_Multi-View_25D_Point_Clouds). We propose a novel architecture named \textit{Edge-Aware PointNet}, that incorporates complementary edge information with the recently proposed PointNet++ framework, by making use of convolutional neural networks (CNNs).

![prediction example](https://github.com/Merium88/Edge-Aware-PointNet/blob/master/doc/method.jpg)

In this repository, we release code and data for training the network Edge-Aware PointNet on point clouds sampled from 3D shapes.

### Usage
The code is written as an extension to the original PointNet++ thus the usage and training procedure is the same as for the original repository. (https://github.com/charlesq34/pointnet2)
To train a model to classify point clouds sampled from ModelNet40:

        python train_modelnet40_edgecnn.py



