> []# EV-SegNet: Semantic Segmentation for Event-based Cameras

## Introduction
  To capture the dynamic intensity changes during events, promising sensors such as event cameras or Dynamic Vision Sensor (DVS) are often adopted. While these cameras provide high temporal resolution, high dynamic range and require low on power and bandwidth, deep learning method still barely use the images of these cameras. The main reasons are 
1. No standard output images of these cameras；
2. No common representation of the stream to feed a CNN；
3. Lack of labelled training data. 

In 2018, Iñigo Alonso and Ana C. Murillo published an article to overcome these two challenges by proposing a semantic segmentation CNN, EV-SegNet, a novel representation for DVS data and also the method for generating semantic segmentation labels. This blog intends to provide readers with insights into these contributions and show our reproduction progress based on these techniques. 

## Related Work 

### Applications of Event Cameras
As mentioned before, event cameras and DVS are barely adopted for deep learning usage. Recent works tend to solve problems (3D reconstruction and 6-DOF camera tracking) with conventional cameras. These tasks already showed mature performance on RGB images, however turn out to be not adaptable for event cameras. In regard to other tasks, such as optical flow estimation, object detection and recognition, latest works start to deal with recordings in real scenarios, such as N-CARS dataset and DDD17 dataset. It's interesting that all these works begin with a common first-step: to encode the event information into an image-like representation. 

### Semantic Segmentation
For a given image, one would expect a segmentation algorithm to recognize the semantic parts in it through a segmentation algorithm$^{[1]}$. Semantic segmentation as the expected solution, mainly aims at assigning a semantic label to each pixel in the image. Most of the works that focus on this visual recognition problem are realized by encoder-decoder CNN architectures. The existing methods for semantic segmentation mostly either target at an instance level or a class level and prove the effecetiveness of CNNs. Based on this, the author managed to find some modalities as addition to the input data and it turns out better performance has been achieved with these modalities added. And specifically in this case where the author intends to adopt event camers, event camera data will be adopted as the additional modality.
## Data
### Event data
Unlike RGB conventional image, which is 3-dimensional (height, width and 3 channels), the output of an event camera does not have an standard representation. Events are asynchronous and some previous encodings of events do not provide good input for CNNs.

### Event Representation
In order to better train a CNN model with event data, the paper proposed an event data representation which is 6 channels. The first two channels are the histogram of positive and negative events. Second two and the last two are the mean and standard deviation of the normalized timestamps of events happening at each pixel for positive and nagative events.Thus the six channels are $Hist(x, y, -1)$, $Hist(x, y, +1)$, $M(x, y, -1)$, $M(x, y, +1)$, $S(x, y, -1)$, $S(x, y, +1)$.

### Dataset 
To train a CNN model with the proposed 6 event  representation, the paper constructed the corresponding dataset with that representation. Its data is an extension for semantic segmentation of the DDD17 dataset, which consists of 40 sequences of diffenrent driving set-ups. The extension first select 6 out of the whole 40 sequences based on two criteria(day-time video and no extreme overexposure) in order to achieve better result. And then it trained another CNN to automatically generate semantic segmentation labels to be used as ground truth. So wrap it up, the X of the dataset is the proposed 6 channels event representation and Y of the dataset is the automatically generated semantic segmentation labels. So next, we are going to introduce the model.
## Model Architecture

| ![Model Figure](https://i.imgur.com/Hsudp2k.png) |
|:--:|
|Figure 1. Segmantic Segmentation from Event-based Cameras|

The model architecture, shown in Fig.1, is inspired on current state-of-the-art semantic segmentation CNNs. While the SOTA semantic segmentation CNNS using RGB data and addtional modalities, this paper slightly adapted to use the event data encodings.

This paper used an encoder-decoder architecture that is commonly used in the related works. Combined with the Xception model as the encoder and a light decoder, the model concentrates the heavy computation on the encoder.

This model architecture of this paper also includes features from the most successful recent models for semantic segmentation, including: skip connections, an auxiliary loss.

Xception is the main training focus, and is also where our biggest problem came from when migrating from Tensorflow to Pytroch. Which we will talk later in this blog. As shown in Fig 2, Xception is a convolutional neural network with 36 layers deep  and an image input size of 299-by-299. It involves Depthwise Seperable Convolutions layers with residual connections. Noted that the input channels in the original paper is 3.



| ![](https://i.imgur.com/C4paM1C.png) |
|:--:|
|Figure 2. Xception architecture  $^{[3]}$    |




## Experiments & Results

### Experimental Environments
| Setting     | Description                       |  
| ----------- | --------------------------------- |  
| System      | Ubuntu 16.04.7 LTS (Xenial Xerus) |  
| GPU         | Nvidia GTX 2080Ti                 |  
| Language    | Python3                           |  
| Packages    | PyTorch1.9, TensorFlow1.13        |  

### Variant 1 - TensorFlow to PyTorch
The author's code was written in TensorFlow v1.13 which is outdated and not very efficient. In our experiment, this version only occupies around 22% voltage utility of one GPU node. Besides, the PyTorch framework is more tightly integrated with Python and we are more family with this. So our first variant is trying to migrate the TensorFlow version to the PyTorch platform. 

The biggest problem we encountered here is the use of pre-trained model. The author used a Keras pre-trained Xception model to get specific layers' output and further use them in the encoder. This pre-trained model can accept the 6-channels input naturally. We have to use a [third party pre-trained model](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py)$^{[2]}$ because the PyTorch didn't provide an official one. This model can only accept the fixed size 3-channels input image, while our project need to feed a 6-channels one. Our first intuitive solution is tryng the 3-channel input first to see what we can get, which means we just use 3 channels of the provided 6-channel dataset. Then we use two Xception models to embed the 6-channels input image spearately inspired by the group normalization.


| ![](https://i.imgur.com/XnlTMvy.png)|
|:--:|
|Figure 3. PyTorch 3-channel Segmentation Result |

As shown in Figure 4, the 6-channels version is even worse than the 3 channel one. Based on our analysis, this combination might change the latent data distribution.

| ![](https://i.imgur.com/9CBil2Y.png =550x400) |
|:--:|
|Figure 4. Comparison between Original Implementation and PyTorch Version. The three results (original-tensorflow version,  pytorch-3channel,  pytorch-6channel) all trained on 10% of the whole training samples.|


### Variant 2 - Hyperparameter Check
For this parameter, we would like to test how is the change of learning rate and the size of dataset would affect the results. 
#### Scaling Down the Dataset
We scale down the dataset for two purpose. First the original dataset is 31G and takes around 70 hours to train the whole dataset using the TensorFlow code. This is really a burden for computation and time comsuming if we want to test the different parameter so we first see if we can use a smaller dataset. The goal is that this scaled down dataset can achieve a similar result.
Secondly, we want to check the robustness of this model. If this model can achieve a relatively decent result with a smaller dataset, then we can prove the validity of the model on a small scale training set. The result is shown in Figure 7.

#### Learning Rate
We reproduce the original tensorflow version on three different learning rate, while one of them is that the authers used. By performing this we want to check if the model is sensitive to the change of learning rate and if the authers used a relatively proper learning rate to implement it. The result is shown in Figure 6.


| ![](https://i.imgur.com/97nOwgO.jpg)|
|:--:|
|Figure 5. Comparison between Results of Original Data and Resized Data.|
|original img(top left) \| gray scale(top right) \| reselt trained on full dataset(bottom left) \| reselt trained on 20% dataset(bottom right)   |




|![](https://i.imgur.com/zdlqITv.png =550x400)|
|:--:|
|Figure 6. Learning Rate Hyperparam Check|


|![](https://i.imgur.com/kBN7Biu.png =550x400)|
|:--:|
|Figure 7. Training Samples Volumn Robustness Check|







## Conclusions

- The model of this paper performs well in terms of metrics like accuracy & IoU, but seems not function well on segmenting an arbitrary RGB img.


## Reference
[1] Zhu, H., Meng, F., Cai, J., & Lu, S. (2016). Beyond pixels: A comprehensive survey from bottom-up to semantic image segmentation and cosegmentation. Journal of Visual Communication and Image Representation, 34, 12-27.
[2] https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
[3] https://arxiv.org/pdf/1610.02357.pdf


## Work Division
| Name              | Work                                                          |  
| ----------------- | ---------------------------------                             |  
| Shiduo Xin        | Run the original code in TensorFlow                           |  
|                   | Integrating and debugging the whole model                     |  
|                   | Hyperparameter Check                                          |  
|                   | Writing Blog (Experiment)                                     |  
| Xinqi Li          | Rewrite TensorFlow code to PyTorch (model)                    |  
|                   | Integrating and debugging the whole model                     |  
|                   | Writing Blog (Experiment)                                     |  
| Dongxu Lu         | Rewrite TensorFlow code to PyTorch (framework)                |  
|                   | Writing Blog (Introduction, Related Work) and organize content|  
| Kexin Su          | Rewrite TensorFlow code to PyTorch (data loader)              |  
|                   | Wriring Blog (Data, Model Architecture)                       |  