# PROBA-V
A solution to the PROBA-V Super Resolution Competition. This solution treats the multiple low resolution images as frames of a 'video' and 3D Convolutional Network to produce a super resolution image from the low resolution ones.

## TODO List
- [x] Preprocessing Pipeline
- [x] Training Framework
- [x] Prediction Framework (for competition submission)
- [ ] Preliminary Report
- [ ] Parse Config Framework
- [ ] Multi-GPU support
- [ ] Colored Images support

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

| Net           | Data          | ResBlocks | Filters  | Loss | Normalization |Score |
| ------------- |:-------------:| -----:| -----:|-----:|-----:|-----:|
| Conv3D + WDSR    | Patches 32x32 70% Clarity 7 LR Images | 8 |32  |L1  | Weight  |-  |
| Conv3D + WDSR      | Patches 38x38  90% Clarity 7 LR Images |   8 | 32    |L1    | Weight  |-    |
| Conv3D + WDSR      | Patches 38x38  90% Clarity 7 LR Images |   10 | 32    |L1    | Weight  |-    |
| Conv3D + WDSR      | Augmented Patches 38x38 85% Clarity 7 LR Images |   10 | 32    |L1    | Weight  |-    |
| Conv3D + WDSR      | Augmented Patches 38x38 85% Clarity 9 LR Images |   10 | 32    |L1    | Weight  |-    |
| Conv3D + WDSR + InstanceNorm     | Augmented Patches 38x38 85% Clarity 9 LR Images |   10 | 32    |L1    | Weight  |-    |

## The Model
* [3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks](https://arxiv.org/abs/1812.09079)
* [Wide Activation for Efficient and Accurate Image Super-Resolution](https://arxiv.org/abs/1808.08718)
* [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
