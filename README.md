# Keras_BASNet
中文版

这个项目主要利用keras实现了如下论文中提出的BASNet。该论文是基于Unet网络结构的改进。BASNet主要通过在损失函数中引入结构相似度评估准则（SSIM），来提升网络模型对图片边缘的注意力，使得分割图片的边缘更加精准。

[原论文](http://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)

[代码【pytorch】](https://github.com/xuebinqin/BASNet)

English vision

This project mainly uses Keras to implement the **BASNet** proposed in the following paper. This paper is based on the improvement of UNET network structure. Basnet mainly introduces Structural Similarity Assessment Criterion (SSIM) into the loss function to improve the attention of the network model to the edge of the image, so as to make the edge segmentation of the image more accurate.

[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)

[code【pytorch】](https://github.com/xuebinqin/BASNet)

#lRequired libraries
keras =  2.2.5
tensorflow = 1.14.0

# Paper Citation
@InProceedings{Qin_2019_CVPR,
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
title = {BASNet: Boundary-Aware Salient Object Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
