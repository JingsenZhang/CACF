# CACF
This is the Pytorch implementation for our CIKM 2021 short paper:
> Jingsen Zhang, Xu Chen, and Wayne Xin Zhao (2021)."Causally Attentive Collaborative Filtering." In CIKM 2021. 

## Overview
We propose to equip attention mechanism with causal inference, which is a powerful tool to identify the real causal effects. Our model is based on the potential outcome framework. In specific, the real causal relation of each feature on the outcome is measured by the individual treatment effect (ITE) and we minimize the distance between the traditional attention weights and the normalized ITE. With such causal regularization, the learned attention weights can reflect the real causal effects.

<img src="https://github.com/JingsenZhang/CACF/blob/master/img/model.png" width = "500px" align=center />

## Requirements
- Python 3.7
- Pytorch >=1.3

Notice: All the models are implemented based on [RecBole](https://github.com/RUCAIBox/RecBole), a popular open-source recommendation framework. 

## Datasets
We use three real-world benchmark datasets, including *MovieLens-100K*, *Amazon-Electronics* and *Book-Crossing*. All the datasets are available at this [link](https://recbole.io/dataset_list.html).

## Usage
+ Download the codes and datasets.
+ Run

  ++ Run run_cacf.py

```
python run_cacf.py --model=CACF --dataset=ml-100k --config_files='cacf.yaml ml-100k.yaml'
```
  ++ the meanings of parameters
```
--model         model name: CACF, Pop, BPR, FM, WideDeep, AFM, DeepFM 
--dataset       dataset name: ml-100k, amazon_electronics, book_crossing
--config_files  the configuration files for model and dataset
```

+ Parameter Settings

The search ranges of some parameters are shown below. You can configure training parameters through the command line.
```
--learning_rate    [0.05, 0.01, 0.005, 0.001]
--embedding_size   [10, 16, 32, 64]
--train_batch_size [256, 512, 1024, 2048]
--mlp_hidden_size  [[32, 16, 8], [64, 32, 16], [128, 64, 32]]
--dropout_prob     [0.0, 0.05, 0.1, 0.2, 0.5]
--t                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
--loss_weight1     [0.1, 0.5, 0.9]
--loss_weight2     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

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
