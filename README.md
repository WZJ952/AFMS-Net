# Adaptive Frequency Enhancement and Multi-Scale Semantic Interaction for Robust Cross-View Geo-Localization

Code for our model.

## Prerequisites

- torch
- torchvision
- numpy 
- pyyaml
- tqdm
- scipy
- matplotlib
- pillow

## Dataset & Preparation
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request and put them under the `./data/` folder. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

## Pretrained Vit-S weights
You can download the pretrained Vit-S weights from the following link and put it in the **./models/pretrain_model** folder

- [Google Driver](https://drive.google.com/file/d/1QQ-KpJJsn-hAzwWx6Lnb5D-U1PhH93Y7/view?usp=sharing)

## Train & Evaluation
### Train & Evaluation on **University-1652**
```
bash total.sh
```
* You can change the **data_dir** and **test_dir** to your own dataset paths in **total.sh**. 

## TO-DO List

- [ ] Support SUES-200 dataset
- [ ] ...

## Reference
- **University-1652**: [pdf](https://arxiv.org/abs/2002.12186)|[code](https://github.com/layumi/University1652-Baseline)

## Request
This code is directly associated with the manuscript currently submitted to The Visual Computer (titled "Adaptive Frequency Enhancement and Multi-Scale Semantic Interaction for Robust Cross-View Geo-Localization"). We kindly encourage the citation of this related work.
