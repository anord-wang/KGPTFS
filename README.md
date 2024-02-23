
# KGPTFS

This ReadMe file contains the Python codes for the KGPTFS [paper](): Knockoff-Guided Feature Selection via A Single Pre-trained Reinforced Agent.

# 1. Task and Solution
Our task is to select features with unsupervised methods including Knockoff and Matrix reconstruction.

Our method involves generating "knockoff" features that replicate the distribution and characteristics of the original features but are independent of the target variable. Each feature is then assigned a pseudo label based on its correlation with all the knockoff features, serving as a novel metric for feature evaluation.

# 2. Dataset
We use several Datasets for experiments. The data is in this [page](https://drive.google.com/file/d/1nQJd2bs7Tb6qykPUhQr4no5O-Ju6A1Q2/view?usp=sharing).

# 3. Codes Description
There are two parts of the code. The first part is the modified Attention code. The second part is the progress of the proposed method.

## 3.1. Data Processing Codes
The data processing code is [knockoff_data_generation.py](knockoff_data_generation.py). It can generate the Knockoff features based on the original features.

## 3.2. Main Codes
The main code is [FRL_Recon_knockoff.py](FRL_Recon_knockoff.py), which illustrates our method. 

The other codes start with 'FRL' are our ablation study codes.
