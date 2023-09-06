# A Non-Local Block with Adaptive Regularization Strategy
Zhonggui Sun, Huichao Sun, Mingzhu Zhang, Jie Li, Xinbo Gao.

This code is based on the [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) from [OpenMMlab](https://openmmlab.com/) 
__________
**Contents**
- [Abstract](#abstract)
- [Brief Introduction](#brief-introduction)
- [Usage](#usage)
- [Results](#results)
  - [Quantitative Results](#quantitative-results)
  - [Qualitative Results](#qualitative-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Abstract
Non-local block (NLB) is a breakthrough technology in computer vision. It greatly boosts the capability of deep convolutional neural networks (CNNs) to capture long-range dependencies. As the critical component of NLB, non-local operation can be considered a network based implementation of the well-known non-local means filter (NLM). Drawing on the solid theoretical foundation of NLM, we provide an innovative interpretation of the non-local operation. Specifically, it is formulated as an optimization problem regularized by Shannon entropy with a fixed parameter. Building on this insight, we further introduce an adaptive regularization strategy to enhance NLB and get a novel non-local block named ARNLB. Preliminary experiments on semantic segmentation demonstrate its effectiveness.

## Introduction
<div align=center><img src="https://github.com/sunhuichao/ARNLB/blob/main/ARNLB.png"/></div>

Our contributions are twofold:
1) We provide a rigorous theoretical interpretation for the
non-local operation in NLB. Specifically, by utilizing a twostep regularization framework, the operation can be seen as
the result of an optimization problem, which is regularized by the Shannon entropy with a fixed parameter.
2) Building upon the interpretation, we further impose an adaptive regularization strategy into the implementation of the
non-local operation and get a novel version of NLB, which we refer to ARNLB. Preliminary experiments on semantic segmentation demonstrate its effectiveness.

## Usage
Please refer to [MMsegmentation](https://mmsegmentation.readthedocs.io/en/latest/) help documentation.

## Result
### Quantitative Results
![image](https://github.com/sunhuichao/ARNLB/blob/main/Table%20I.png)
### Qualitative Results
<div align=center><img src="https://github.com/sunhuichao/ARNLB/blob/main/Fig%202.png"/></div>

## Acknowledgments
The authors would like to express their great thankfulness to the Associate Editor and the anonymous reviewers for
their valuable comments and constructive suggestions. At the same time, they would like to express their sincere gratitude to the open-source semantic segmentation library MMSegmentation from openmmlab.
## Citation
