# CACF
This is the Pytorch implementation for our CIKM 2021 short paper:
> Jingsen Zhang, Xu Chen, and Wayne Xin Zhao (2021)."Causally Attentive Collaborative Filtering." In CIKM 2021. 

## Overview
We propose to equip attention mechanism with causal inference, which is a powerful tool to identify the real causal effects. Our model is based on the potential outcome framework. In specific, the real causal relation of each feature on the outcome is measured by the individual treatment effect (ITE) and we minimize the distance between the traditional attention weights and the normalized ITE. With such causal regularization, the learned attention weights can reflect the real causal effects.

<img src="https://github.com/JingsenZhang/CACF/blob/master/model.png" width = "500px" align=center />

## Requirements
- Python 3.7
- Pytorch >=1.3

Notice: All the models are implemented based on [RecBole](https://github.com/RUCAIBox/RecBole), a popular open-source recommendation framework. 

## Datasets
We use three real-world benchmark datasets, including *MovieLens-100K*, *Amazon-Electronics* and *Book-Crossing*. All the datasets are available at this [link](https://recbole.io/dataset_list.html).

## Results

## Usage
### Download the code and dataset
### Run

### Parameter Settings

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
````
@inproceedings{Zhang-CIKM-2021,
    title = "Causally Attentive Collaborative Filtering",
    author = {Jingsen Zhang and
              Xu Chen and
              Wayne Xin Zhao},
    booktitle = {{CIKM}},
    year = {2021},
}
````
If you have any questions for our paper or codes, please send an email to zhangjingsen@ruc.edu.cn.
