# bnw2color
CS 663 (Digital Image Processing) Project

## Team
#### Rajat Rathi (160050015)
#### Anmol Singh (160050107)
#### Gurparkash Singh (160050112)

## Overview

The problem we are trying to solve is a general conversion of black and white images to colored. For this, we will be using **Conditional GANs** [Generative Adversarial Networks] to implement and extend the **pix2pix** network, which is a general neural network model to learn a mapping from one set of images to another, which, in our case will be from the set of black and white images to colored images.

Our project can be easily extended to other tasks such as day to night, labels to street scene, deblurring images etc. as the basic pix2pix network is shared among all such applications. If time permits, we will try to apply our network to one of the other tasks as well. We will be using the PyTorch framework in Python to implement the network.

## Research Paper
### Image-to-Image Translation with Conditional Adversarial Networks [1]

This is the original Research Paper written on the pix2pix network written at the Berkeley AI Research (BAIR) Laboratory, UC Berkeley. This is the research paper that we are trying to replicate and apply to the task of colorizing Black and White images.

## Datasets
### CIFAR-10 [2]

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

### Places365 [3]

In total, the Places dataset contains more than 10 million images comprising 400+ unique scene categories. The dataset features 5000 to 30,000 training images per class, consistent with real-world frequencies of occurrence.

### Microsoft COCO Dataset [4]

It is a publicly available dataset of images with their captions. We can use a subset of those images to train and  test the images.

## Evaluation Metrics
For evaluation we are using the same metrics as used in the original paper (stated below). We will be comparing our results with that of the original research paper.
### Per-pixel accuracy
### Per-class accuracy
### Class IOU

## References:

<ol>
	<li>pix2pix Research Paper: https://arxiv.org/pdf/1611.07004.pdf</li>
	<li>CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html</li>
	<li>Places-365 Dataset: http://places2.csail.mit.edu/</li>
	<li>Microsoft COCO: http://cocodataset.org/</li>
	<li>Official pix2pix Repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix</li>
	<li>PyTorch Website: https://pytorch.org/</li>
</ol>

## Tutorials:
<ul>
	<li>Intro to GANs: https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394</li>
	<li>Generative Models (Stanford): https://www.youtube.com/watch?v=5WoItGTWV54&index=13&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv</li>
	<li>GANs in PyTorch (Simple): https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f</li>
	<li>GANs in PyTorch (Official): https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html</li>
	<li>pix2pix Tutorial: https://towardsdatascience.com/cyclegans-and-pix2pix-5e6a5f0159c4</li>
	<li>Colorization with GAN Repo: https://github.com/ImagingLab/Colorizing-with-GANs</li>
	<li>GAN Hacks Repo: https://github.com/soumith/ganhacks</li>
	<li>Installing CUDA 9.0: https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8</li>
	<li>PyTorch Examples: https://cs230-stanford.github.io/pytorch-getting-started.html</li>
</ul>