# Neoadjuvant Chemotherapy Simulation for Breast Cancer MRI via Diffusion Model

 [![arXiv](https://img.shields.io/badge/ISBI2025-10981225-f9f107.svg?style=plastic)](https://ieeexplore.ieee.org/abstract/document/10981225) [![arXiv](https://img.shields.io/badge/arXiv-2509.24185-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2509.24185v1)

## Overview

This repository contains the code for Neoadjuvant Chemotherapy Simulation for Breast Cancer MRI via Diffusion Model. The model architecture is illustrated below: 

![fig2](./asset/fig2.png)



Our code was written by applying ControlNet. We would like to thank those who have shared their code.

- [Adding Conditional Control to Text-to-Image Diffusion Models](https://github.com/lllyasviel/ControlNet?tab=readme-ov-file). 



## Enviroment

First create a new conda environment

```shell
conda env create -f environment.yaml
conda activate NACsim
```



## Pretrained model

All pretrained ControlNet can be downloaded from [Hugging Face page](https://huggingface.co/thibaud/controlnet-sd21/tree/main). 



## Train

```shell
python train.py
```



## Citation

```tex
@inproceedings{kim2025simulating,
  title={Simulating Post-Neoadjuvant Chemotherapy Breast Cancer MRI via Diffusion Model with Prompt Tuning},
  author={Kim, Jonghun and Park, Hyunjin},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

