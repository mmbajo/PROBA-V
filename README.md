# PROBA-V
A solution to the PROBA-V Super Resolution Competition. This solution treats the multiple low resolution images as frames of a 'video' and 3D Convolutional Network to produce a super resolution image from the low resolution ones.
## Requirements
```python
torch=1.4
tensorflow=2.0.1
tensorflow-addons=0.5.2
scikit-image=0.15
numpy
matplotlib
tqdm
```

## Usage
I shall implement an editable config file mechanism in the future. I find it really annoying when I make a typo in command line and not being able to correct it fast.
### Preprocessing
```sh
python3 utils/dataGenerator.py --dir probav_data \
                               --ckptdir dataset \
                               --band NIR

```
### Train
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
Here are what I tried. Most of them did not end well. I am still waiting for the result of my submissions. But I believe that my final network achieved good results.

| Net           | Data          | Blocks | Filters  | Loss | Normalization |Score |
| ------------- |:-------------:| -----:| -----:|-----:|-----:|-----:|
| Conv3D + WDSR    | Patches 32x32 | 8 |32  |L1  | -  |-  |
| Conv3D + WDSR      | Patches 38x38   |   8 | 32    |L1    | Weight  |-    |
| Conv3D + WDSR      | Patches 38x38   |   10 | 32    |L1    | Weight  |-    |
| Conv3D + WDSR      | Augmented Patches 38x38   |   10 | 32    |L1    | Weight  |-    |

## The Model
* [3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks](https://arxiv.org/abs/1812.09079)
* [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718)
* [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)

## Understanding the problem
The challenge is to restore a low resolution image to a high resolution one. Simple enough?? Let's go!

## Preliminary thoughts on how to solve this problem
* Ok I am a noob so do not judge me. (Ok go on. I know this repository is meant to judge me.)
* Upon, looking into the problem, I can imagine using k-nearest neighbors like techniques to estimate high resolution pixels from the low resolution ones.
* Random forest might be a good idea too but I believe compare to deep CNNs these techniques won't even be a contender!!
* First, I am thinking of using a convolutional neural net to input the low resolution images and spit out some high resolution images. Let's start with naive and simple solutions!
* Second, I am thinking of using an ensemble of neural nets! You know the saying if one doctor is not enough consult several doctors at once. Well, there are no saying such as those lines I have typed. But yeah, let's try an ensemble. This will be my first time so it will be a fun learning experience.
* Third, OK, at this point in time I don't know what solutions are viable so I would like to consult Dr. Google and Prof. arXiv.

Anyway, I received the challenge today Feb. 07, 2020 and started writing my thoughts this day... duh. What am I even saying!
## TODO:
* Put in hell of a lot of figures cuz I am such a noob at writing and talking about anything. They say that an image is a worth a trillion words... damn I am really slaying this challenge with damn sayings.
* Reread the problem again and update The understanding the problem part. Cuz right now your description is such a noooob one. Remember not to be stagnant on one place! And please check your english!
* Once the problem is reread, think of how you would define the loss...
* Let's go python. Search how the top solutions are made. and maybe combine them?? Cuz I am not creative enough to form my own solution. Well, I am kidding you. I can do it given infinite amount of time and ... GPU aalkdfjlasjdflaskjdflaksjfd
* This challenge is harder than I thought. I need to read about Image Registration and stuff... my goal for the next round is to code the loss function...
