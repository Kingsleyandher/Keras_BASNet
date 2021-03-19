# Keras_BASNet
# 中文版

这个项目主要利用keras实现了如下论文中提出的**BASNet**。该论文是基于Unet网络结构的改进。BASNet主要通过在损失函数中引入结构相似度评估准则（SSIM），来提升网络模型对图片边缘的注意力，使得分割图片的边缘更加精准。

[原论文](http://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)

[代码【pytorch】](https://github.com/xuebinqin/BASNet)  

# 效果展示
原始图片：  

<div align=center><img src="https://github.com/Kingsleyandher/Keras_BASNet/blob/main/figure/original.png" width="300" height="450" /></div>

Unet-RestNet50 分割结果：  

<div align=center><img src="https://github.com/Kingsleyandher/Keras_BASNet/blob/main/figure/Unet_resnet50.png" width="300" height="450" /></div>

Unet-VGG16 分割结果：  

<div align=center><img src="https://github.com/Kingsleyandher/Keras_BASNet/blob/main/figure/Unet_vgg16.png" width="300" height="450" /></div>

BASNet-ResNet34 分割结果：

<div align=center><img src="https://github.com/Kingsleyandher/Keras_BASNet/blob/main/figure/BASNet.png" width="300" height="450" /></div>


# 注意
说明：目前仅支持在图片中提取单个种类的目标，因此分类种类即前景和背景总共两类。  

# 训练自己的数据

训练集：训练图片 训练标签  
分别放在```/Data/train/train_image/``` 和 ```/Data/train/train_label/``` 路径下。  
注意必须保持同一张图片和标签名称的一致性，且最好均为png文件。  
测试集：测试图片 测试标签
分别放在```/Data/test/test_image/``` 和 ```/Data/test/test_label/``` 路径下。  
注意必须保持同一张图片和标签名称的一致性，且最好均为png文件。  



# English vision

This project mainly uses Keras to implement the **BASNet** proposed in the following paper. This paper is based on the improvement of UNET network structure. Basnet mainly introduces Structural Similarity Assessment Criterion (SSIM) into the loss function to improve the attention of the network model to the edge of the image, so as to make the edge segmentation of the image more accurate.

[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)

[code【pytorch】](https://github.com/xuebinqin/BASNet)

# Required libraries

keras =  2.2.5  

tensorflow = 1.14.0

# Notice

Currently only the target of extracting a single category in the image is supported, so the category category is a total of two categories, namely foreground and background.  


# Train your own data

Training set: Training images and Training labels  

```/Data/train/train_image/``` and ```/Data/train/train_label/```   respectively.  

Note that the same image and label name must be consistent, preferably in a PNG file.  

Test set: Testing image and Testing label  

```/Data/test/test_image/``` and ```/Data/test/test_label/```, respectively.  

Note that the same image and label name must be consistent, preferably in a PNG file.  


# Paper Citation

> ```
> @InProceedings{Qin_2019_CVPR,
> author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
> title = {BASNet: Boundary-Aware Salient Object Detection},
> booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
> month = {June},
> year = {2019}
> }
> ```


