# Training and Evaluation of Deep Learning Models for Quantitative Disease Assessment using High-Resolution Wheat Canopy Imagery

This repository contains the code, configuration files, and software environment used to train and evaluate the deep learning models presented in **From Canopy Images to Foliar Disease Assessments: A Scalable Approach to Measure Quantitative Resistance in the Field**.

TODO: Add doi

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <https://github.com/and-jonas/train-eval-leaf>
cd train-eval-leaf
pip install -r requirements.txt
```

## Reproducing the experiments

Experiment configurations are provided in the `config/` directory, including model architectures, training parameters, data augmentation, and dataset settings.

Experiments can be executed using the corresponding configuration files, for example:

```bash
python train_eval_seg.py \
    --config-name experiment/05_seg_org_architecture
```

Dataset paths and pretrained model weight locations must be adapted to the local environment.

## Data and pretrained models

Refer to Zenkl et al. 2026 TODO Add doi 
