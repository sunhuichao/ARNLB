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

### Preparation

