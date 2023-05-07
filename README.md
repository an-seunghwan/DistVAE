# Distributional Variational AutoEncoder: Application to Synthetic Data Generation

This repository is the official implementation of Distributional Variational AutoEncoder: Application to Synthetic Data Generation (DistVAE) with pytorch. 

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Package Dependencies

```setup
python==3.7
numpy==1.21.6
torch==1.13.0
```
Additional package requirements for this repository are described in `requirements.txt`.

## Training & Evaluation 

### 1. How to Training & Evaluation  

#### training
```
python main.py --dataset <dataset>
```   

#### evaluation
- Synthetic data generation and evaluation: `synthesize.py`
- CDF estimation and Vrate: `inference.py`
- Discretize estimated CDF example: `calibration.py`
- Membership inference attack for VAE: `shadow_data.py`, `shadow_main.py`, `shadow_attack.py`

## Results

<center><img  src="https://github.com/an-seunghwan/DistVAE/blob/main/assets/cabs/cabs_estimated_quantile.png?raw=true" width="800"  height="150"></center>

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
+-- requirements.txt
+-- LICENSE
+-- README.md
```