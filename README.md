# CACF
This is the Pytorch implementation for our CIKM 2021 short paper:
> Jingsen Zhang, Xu Chen, and Wayne Xin Zhao (2021)."Causally Attentive Collaborative Filtering." In CIKM 2021. 

## Overview
We propose to equip attention mechanism with causal inference, which is a powerful tool to identify the real causal effects. Our model is based on the potential outcome framework. In specific, the real causal relation of each feature on the outcome is measured by the individual treatment effect (ITE) and we minimize the distance between the traditional attention weights and the normalized ITE. With such causal regularization, the learned attention weights can reflect the real causal effects.

<img src="https://github.com/JingsenZhang/CACF/blob/master/model.png" width = "500px" align=center />

## Requirements
- Python 3.7
- Pytorch 1.7.1
- CUDA 11.0

Notice: All the models are implemented based on [RecBole](https://github.com/RUCAIBox/RecBole), a popular open-source recommendation framework. 

## Datasets
## Parameter Settings
## Usage
## Acknowledgement
