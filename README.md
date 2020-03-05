# EnhanceMe! : PROBA-V Super Resolution Challenge
<p align="center"> <img src="img/lr2hr.png"> </p>

A solution to the PROBA-V Super Resolution Competition. This solution treats the multiple low resolution images as frames of a 'video' and 3D Convolutional Network to produce a super resolution image from the low resolution ones.


## TODO List
- [x] Preprocessing Pipeline
- [x] Training Framework
- [x] Prediction Framework (for competition submission)
- [ ] Preliminary Report
- [ ] Code cleanup
- [ ] Low resource training backend
- [ ] Parse Config Framework
- [ ] Multi-GPU support
- [ ] Colored Images support

## Setup to get started
```python
pip3 install -r requirements.txt
```

## Usage
I shall implement an editable config file mechanism in the future as I find it really annoying when I make a typo in command line and not being able to correct it fast. But maybe one could use shell scripts for this.
### Preprocessing
```sh
python3 utils/dataGenerator.py --dir probav_data \
                               --ckptdir dataset \
                               --band NIR

```
### Train

The training was done in a computer with the following specifications:
* RAM: **64GB**
* Swap Space: **72GB**
* GPU: **GTX1080ti**
If you don't have a computer with high RAM, consider lowering the batch size or lowering the number of residual blocks of the network. If you have better specs, try raising the number of low resolution images and increasing the residual blocks for better performance.

```sh
python3 train.py --data dataset/augmentedPatchesDir \
                 --band NIR \
                 --split 0.3 \
                 --batchSize 64 \
                 --epochs 100 \
                 --logDir modelInfo/logs \
                 --ckptDir modelInfo/ckpt \
                 --optim nadam \

```
### Test
```sh
python3 test.py --data dataset/augmentedPatchesDir \
                --band NIR \
                --modelckpt modelInfo/ckpt \
                --output output
```

## The Results
Here are what I tried. Most of them did not end well. I am still waiting for the result of my submissions. (The Post-Mortem evaluation server is down at the moment.)

| Net           | Data          | ResBlocks | Filters  | Loss | Normalization |Hold Out Test Score | Competition Score |
| ------------- |:-------------:| -----:| -----:|-----:|-----:|-----:|-----:|
| Conv3D + WDSR    | Patches 32x32 70% Clarity 7 LR Images | 8 |32  |L1  | Weight  |~1.456 |-  |
| Conv3D + WDSR      | Patches 38x38  90% Clarity 7 LR Images |   8 | 32    |L1    | Weight  |~1.345  |-  |
| Conv3D + WDSR      | Patches 38x38  90% Clarity 7 LR Images |   10 | 32    |L1    | Weight  |~1.306   |-  |
| Conv3D + WDSR      | Augmented Patches 38x38 85% Clarity 7 LR Images |   10 | 32    |L1    | Weight  |~1.304 |-  |
| Conv3D + WDSR      | Augmented Patches 38x38 85% Clarity 9 LR Images |   10 | 32    |L1    | Weight  |~1.230 |-  |
| Conv3D + WDSR  | Augmented Patches 38x38 85% Clarity 9 LR Images |   10 | 32    |L1+SobelL1 Mix   | Weight  |~1.232   |-  |
| Conv3D + WDSR  | Augmented Patches 38x38 85% Clarity 9 LR Images |   12 | 32    |L1+SobelL1 Mix   | Weight  |~1.225    |-  |
| Conv3D + WDSR + InstanceNorm     | Augmented Patches 38x38 85% Clarity 9 LR Images |   12 | 32    |L1+SobelL1 Mix  | Weight  |-    |-  |
| Conv3D + WDSR + InstanceNorm     | Augmented Patches 38x38 85% Clarity 9 LR Images |   12 | 64    |L1+SobelL1 Mix  | Weight  |-    |-  |

Note: Lower is better.

## The Preprocessing
The preprocessing steps are the following:
* Filtering out data sets with all its LR images contain clarity below 85%.
* Picking out k best LR images.
* Registering the LR images using this [technique](https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html). I used the clearest of the LR images as the reference frame.
* Padding the LR images with additional 3 pixels per side. The official scoring involves cropping the image by 3 pixels all side. This is to compensate for that.
* Patching the LR images to 38x38 sizes and the corresponding HR images to 96x96 patches.
* Removing HR patches with clarity below 85%.
* Augmenting the data set by flipping.
* Augmenting the data set by rotating by 90 to 270 degrees with interval of 90.
* Augmenting the data set by shuffling the LR patches.

## The Model
The model is based on the well known [WDSR](https://arxiv.org/abs/1808.08718) super resolution neural network architecture which performed very good in DIV2K super resolution dataset. This architecture takes in low resolution images and predicts its high resolution version by using 2D convolutional neural network.

PROBA-V dataset is peculiar since multiple low resolution images are available for predicting the high resolution image. We can view this as the temporal information being available to us. In other words, those low resolution images can be treated as frames of a video and in videos, time one dimension of information.

There is this paper where the researchers used [3D Convolutional Residual Networks(3DSRnet)](https://arxiv.org/abs/1812.09079) networks to generate super resolution video from low resolution ones. We will use that architecture along with [WDSR](https://arxiv.org/abs/1808.08718) blocks to build our network.

### Residual Conv3D and WDSR Combined
The proposed architecture in [3DSRnet](https://arxiv.org/abs/1812.09079) is as follows.

<p align="center"> <img src="img/3DSRnet.png"> </p>

Like any residual nets, this architecture has a main path and a residual path. We replace the bicubic upsampling block with Weight normalized Conv2D net of the mean of the low resolution images. We replace the 3D-CNN block with multiple [WDSR](https://arxiv.org/abs/1812.09079) Residual blocks.

<p align="center"> <img src="img/wdsr-b-block.png"> </p>

We also apply [instance normalization](https://arxiv.org/abs/1607.08022) on the images before entering the main and residual paths. In the [DeepSUM](https://arxiv.org/abs/1907.06490) paper, they have used instance normalization in their architecture to make the network training as independent as possible per imageset. Here, we shall see the effect of using instance normalization.

<p align="center"> <img src="img/normalizations.png"> </p>


## The Loss Function
The loss function is a way of expressing what you want the neural net to learn. In my past attempts on this problem, I noticed that the edges of my prediction are not as sharp as that of the high resolution images. So I created a loss function that allows me to penalize the network if my prediction's edges does not match that of the ground truth.

I propose the following loss function.

<p align="center"> <img src="img/loss2.gif"> </p>

where p is the loss mixing hyperparameter which ranges from 0 to 1. This loss mixes the L1 Loss and the sobel L1 loss (difference of the sobel edges of the ground truth and the predicted image). This loss penalizes the network explicitly for not producing sharp edges in the super resolution image.


<p align="center"> <img src="img/sobelloss.png"> </p>


More concretely, we minimize the absolute difference between the sobel filtered predicted(middle) and truth(right) images along with the absolute difference between the unfiltered ones.

* [3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks](https://arxiv.org/abs/1812.09079)
* [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718)
* [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
