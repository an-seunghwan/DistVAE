# Distributional Variational AutoEncoder: Application to Synthetic Data Generation

This repository is the official implementation of Distributional Variational AutoEncoder: Application to Synthetic Data Generation (DistVAE) with pytorch. 

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Training & Evaluation 

### 1. How to Training & Evaluation  

#### training
```
python main.py --dataset <dataset>
```   

#### evaluation (step-by-step)
- Synthetic data generation and evaluation: `synthesize.py`
- CDF estimation and Vrate: `inference.py`
- Discretize estimated CDF example: `calibration.py`
- Membership inference attack for VAE: 
  - `shadow_data.py`: generate training/test datasets for shadow models
  - `shadow_main.py`: training shadow models using training datasets from `shadow_data.py`
  - `shadow_attack.py`: evaluate membership inference attack for VAE

## directory and codes

```
.
+-- assets (for each dataset)
+-- modules 
|       +-- adult_datasets.py
|       +-- cabs_datasets.py
|       +-- covtype_datasets.py
|       +-- credit_datasets.py
|       +-- kings_datasets.py
|       +-- loan_datasets.py
|       +-- evaluation.py
|       +-- model.py
|       +-- simulation.py
|       +-- train.py

+-- main.py
+-- synthesize.py
+-- inference.py
+-- calibration.py
+-- shadow_data.py
+-- shadow_main.py
+-- shadow_attack.py
+-- LICENSE
+-- README.md
```
