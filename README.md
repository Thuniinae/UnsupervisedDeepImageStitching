# Unsupervised Deep Image Stitching: Reconstructing Stitched Images from Feature to Pixel (paper)


## Abstract
Traditional feature-based image stitching technologies rely heavily on the quality of feature detection, often failing to stitch images with few features or at low resolution. The recent learning-based methods can only be supervisedly trained in a synthetic dataset instead of the real dataset, for the stitched labels under large parallax are difficult to obtain. In this paper, we propose the first unsupervised deep image stitching framework that can be achieved in two stages: unsupervised coarse image alignment and unsupervised image reconstruction. In the first stage, we design an ablation-based loss to constrain the unsupervised deep homography network, which is more suitable for large-baseline scenes than the existing constraints. Then, a transformer layer is implemented to align the input images in the stitching-domain space. In the next stage, we design an unsupervised image reconstruction network consist of low-resolution deformation branch and high-resolution deformation branch to learn the deformation rules of image stitching and enhance the resolution of stitched results at the same time, eliminating artifacts by reconstructing the stitched images from feature to pixel. Also, a comprehensive real dataset for unsupervised deep image stitching, which can be availabel at www.github.com/nie-lang/UnsupervisedDeepImageStitching, is proposed to evaluate our algorithms. Extensive experiments well demonstrate the superiority of our method over other state-ofthe-art solutions.

## Dataset for unsupervised deep image stitching


## Experimental results on robustness
