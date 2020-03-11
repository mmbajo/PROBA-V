# EnhanceMe! : PROBA-V Super Resolution Challenge
<p align="center"> <img src="img/mygif3.gif"> </p>

A solution to the [PROBA-V Super Resolution Competition](https://kelvins.esa.int/proba-v-super-resolution/home/) that got **top 2** with a score of **0.9416** in this [leaderboard](https://kelvins.esa.int/proba-v-super-resolution-post-mortem/leaderboard/).


## TODO List
- [x] Preprocessing Pipeline
- [x] Training Framework
- [x] Prediction Framework (for competition submission)
- [x] Parse Config Framework
- [x] Preliminary Report
- [ ] Bicubic Mean Technique comparison graph
- [ ] Code cleanup
- [ ] Low resource training backend
- [ ] Multi-GPU support
- [ ] Colored Images support

## Setup to get started
```python
pip3 install -r requirements.txt
```

## Usage
You should create a new cfg file with the format below. The current repository has a model in it. Should you wish to see the super resolution version of the dataset using my pretrained model, just run the preprocessing script and after that the test.py script and you are good to go. You must first download the data [here](https://kelvins.esa.int/proba-v-super-resolution/data/) and specify the directory of the raw data in the cfg file.
### Configuration (cfg) file
```sh
[Directories]
raw_data=probav_data      <Directory of the raw data.>
preprocessing_out=output  <Directory of where you want to put the output.>
model_out=modelInfo       <Directory of where you want to put the model checkpoints.>
train_out=trainout        <Directory of where you want to put predictions for the train data.>
test_out=testout          <Directory of where you want to put predictions for the test data.>

[Train]
batch_size=128            <Batch size for training.>
epochs=100                <Number of epochs for training.>
learning_rate=0.0005      <Number of epochs for training.>
optimizer=nadam           <Optimizer to use for training.>
loss=l1                   <Loss to use for training.>
split=0.2                 <Validation split.>

[Net]
num_res_blocks=12         <Number of residual blocks for the model graph.>
num_low_res_imgs=9        <Number of low resolution images to pick for the model.>
scale=3                   <Upscale factor of the low resolution image to high resolution image.>
num_filters=32            <Number of filters to use for each residual block.>
kernel_size=3             <Kernel size of the convolution filter.>
exp_rate=8                <Expansion rate of the expansion block inside the residual block.>
decay_rate=0.8            <Decay rate of the decay block inside the residual block.>
is_grayscale=1            <Is the input image grayscale? 1. If not, 0.>

[Preprocessing]
max_shift=6               <Maximum possible shift to account in loss computation and model building.>
patch_size=16             <Base patch size for each low resolution image.>
patch_stride=16           <Base patch size for each low resolution image.>
low_res_threshold=0.85    <Clarity Threshold for the low resolution images.>
high_res_threshold=0.85   <Clarity Threshold for the high resolution images.>
num_low_res_permute=19    <Augment data by permuting the order of the low resolution images.>
to_flip=0                 <Augment data by flipping the images.>
to_rotate=0               <Augment data by rotating the images.>
ckpt=1,2,3,4,5            <Preprocessing checkpoints for debugging.>

```
### Preprocessing
```sh
python3 utils/dataGenerator.py --cfg cfg/p16t9c85r12.cfg \
                               --band NIR

```
### Train

The training was done in a computer with the following specifications:
* RAM: **64GB**
* Swap Space: **72GB**
* GPU: **GTX1080ti**

If you don't have a computer with high RAM, consider lowering the batch size or lowering the number of residual blocks of the network. If you have better specs, try raising the number of low resolution images and increasing the residual blocks for better performance.

```sh
python3 train.py --cfg cfg/p16t9c85r12.cfg \
                 --band NIR

```
### Test
```sh
python3 test.py --cfg cfg/p16t9c85r12.cfg \
                --band NIR
```

## The Results
Here are what I tried. Most of them did not end well. **Posted below is the official score I got from the website.**

| Net           | Data          | ResBlocks | Filters  | Loss | Score |
| ------------- |:-------------:| -----:| -----:|-----:|-----:|
| Conv3D + WDSR    | Patches 32x32 70% Clarity 7 LR Images | 8 |32  |L1  |1.130 |
| Conv3D + WDSR      | Patches 38x38  90% Clarity 7 LR Images |   8 | 32    |L1    |0.9653  |
| Conv3D + WDSR      | Augmented Patches 38x38 85% Clarity 9 LR Images |   10 | 32    |L1    |0.9697	 |
| Conv3D + WDSR  | Augmented Patches 38x38 85% Clarity 9 LR Images |   10 | 32    |SobelL1Mix   |0.9728   |
| Conv3D + WDSR  | Augmented Patches 38x38 85% Clarity 9 LR Images |   12 | 32    |L1  |0.9591   |
| Conv3D + WDSR  | Augmented Patches 22x22 85% Clarity 9 LR Images |   8 | 32    |L1  |0.9550   |
| Conv3D + WDSR  | Augmented Patches 22x22 85% Clarity 9 LR Images |   12 | 32    |L1  |0.9536   |
| Conv3D + WDSR  | Augmented Patches 22x22 85% Clarity 9 LR Images plus LR patch filter of 85% Clarity |   12 | 32    |L1  |0.9416   |


Note: Lower is better.

## Preprocessing Steps
The preprocessing steps are the following:
* Filtering out data sets with all its LR images contain clarity below 85%.
* Picking out k best LR images.
* Registering the LR images using this [technique](https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html). I used the clearest of the LR images as the reference frame.
* Padding the LR images with additional 6 pixels per side which represent the maximum pixel shift that is compensated by the [scoring](https://kelvins.esa.int/proba-v-super-resolution/scoring/).
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

## Some remarks and ideas to try
* Training the data in patches and reconstructing it by just putting the prediction for respective patches might be a hindrance in achieving a good super resolution image. It might be a good idea if we implement a fusing/stitching network for this purpose.
* Same with the registration of the images, it might be a good idea to use a neural net for this task.
* In this implementation, I haven't clipped the LR images according to its bit depth which is 14-bit. Doing so might improve performance since it will remove the outliers in the dataset.
* In reconstructing the test set, applying a sliding window along the LR images sorted from clearest to dirtiest then creating SR images and applying a weighted average of the SR images might create a better prediction for the HR image.
* How about using boosting along with deep neural nets?


## References
I learned a lot from these awesome authors of these research papers and repositories!

* [Super-Resolution of PROBA-V Images Using Convolutional Neural Networks](https://arxiv.org/pdf/1907.01821.pdf)
* [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
* [3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks](https://arxiv.org/abs/1812.09079)
* [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718)
* [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
* [DeepSUM: Deep neural network for Super-resolution of Unregistered Multitemporal images](https://arxiv.org/abs/1907.06490)
* [WDSR Tensorflow implementation by krasserm](https://github.com/krasserm/super-resolution)
* [DeepSUM source code by team Superpip](https://github.com/diegovalsesia/deepsum)
* [3DWDSRnet by frandorr](https://github.com/frandorr)
* [HighRes-net by team Rarefin](https://github.com/ElementAI/HighRes-net)
