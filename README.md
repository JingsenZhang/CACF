# CACF
This is the Pytorch implementation for our CIKM 2021 short paper:
> Jingsen Zhang, Xu Chen, and Wayne Xin Zhao (2021)."Causally Attentive Collaborative Filtering." In CIKM 2021. 
## Overview
We propose to equip attention mechanism with causal inference, which is a powerful tool to identify the real causal effects. Our model is based on the potential outcome framework. In specific, the real causal relation of each feature on the outcome is measured by the individual treatment effect (ITE) and we minimize the distance between the traditional attention weights and the normalized ITE. With such causal regularization, the learned attention weights can reflect the real causal effects.
[](CACF/model.pdf)
## Requirements
## Datasets
## Parameter Settings
## Usage
## Acknowledgement
