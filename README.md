# FDA-TTT: Fourier Domain Adaptation-Test-Time Training for Multi-Center Medical Image Segmentation
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.13+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

## Description
FDA-TTT (Fourier Domain Adaptation - Test Time Training) is a method designed to improve the accuracy of medical image segmentation across multiple centers with varying device data distributions. Traditional Test-Time Adaptation (TTA) techniques use a fixed learning rate for all test samples, which fails to account for complex variations in test data. Our approach introduces a dynamic method for aligning source and target domains at test time.

FDA-TTT utilizes a "Style Cues Bank"—a memory bank that stores Fourier low-frequency components from past test data. By using this bank to identify discrepancies in new test images, we can adaptively blend local low-frequency components, effectively aligning the test image with the source domain. This enhances the segmentation model’s predictions, producing pseudo-labels with higher confidence.

Furthermore, we incorporate a dynamic learning rate into the model by calculating the Kullback-Leibler (K-L) divergence between the pseudo-label and the initial prediction. This adaptive learning rate allows for continuous updates to the model parameters during testing, improving segmentation accuracy in real-time.

Our method, implemented with the U-Net architecture, outperforms existing TTA approaches, demonstrating superior results in multi-center medical image segmentation tasks across two different imaging modalities.


![input and output for a random image in the test dataset](./pictures/main.png)



## Quick start


1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

1. Download the data and run training:
```bash
python train.py 
```





## Usage


### Training
`python predict.py `



### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images.


To predict a multiple images and show them without saving them:



```console
python predict.py 
--i="EyeCrossDataset\images"
--o="EyeCrossDataset\outputs"
--g="EyeCrossDataset\masks"
--model="model.pth"
```


You can specify which transfer learning method to use with:

tent: `--method="tent"`



## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.


## Pretrained model
Coming soon



## Results





![network architecture](./pictures/Results.png)
