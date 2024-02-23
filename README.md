
# KGPTFS

This ReadMe file contains the Python codes for the KGPTFS[paper](): Knockoff-Guided Feature Selection via A Single Pre-trained Reinforced Agent.

# 1. Task and Solution
Our task is to select features with unsupervised methods including Knockoff and Matrix reconstruction.
our method involves generating "knockoff" features that replicate the distribution and characteristics of the original features but are independent of the target variable. Each feature is then assigned a pseudo label based on its correlation with all the knockoff features, serving as a novel metric for feature evaluation.

# 2. Dataset
We use several Datasets for experiments. The data is in this [page](https://drive.google.com/file/d/1nQJd2bs7Tb6qykPUhQr4no5O-Ju6A1Q2/view?usp=sharing).

# 3. Codes Description
There are two parts of the code. The first part is the modified Attention code. The second part is the progress of the proposed method.

## 3.1. Modified Attention Codes
The modified Attention code is in the [folder](modified_transformer/). You can put it in the Transformers lib and the path to those two codes may be like this:

```bash
'/home/local/ASURITE/xwang735/anaconda3/envs/LLM/lib/python3.12/site-packages/transformers/models/gpt2'
```

Or you can just create a new lib containing these codes and name it 'newTransformers'.

## 3.2. Main Codes
There are data preprocessing, pre-training, fine-tuning, and prediction codes in the [src/](src/). 

### 3.2.1. Data Pre-processing Codes
First, the data preprocessing codes contain [data_preprocess_amazon.py](src/data_preprocess_amazon.py), [data_preprocessing.py](src/data_preprocessing.py), and [data_pkl.py](src/data_pkl.py).

[data_preprocess_amazon.py](src/data_preprocess_amazon.py) is used to transform raw data to the format we want. The processed data can be found at this [link](https://drive.google.com/file/d/1LyT0A1NoFJljg8eVWvkoGNvyeOGwX_K9/view?usp=sharing).

[data_preprocessing.py](src/data_preprocessing.py) is used to get the relationship matrix for every dataset.

[data_pkl.py](src/data_pkl.py) is used to get 2-order connection among items.

### 3.2.2. Useful Components
These codes are in [libs/](src/libs/). These codes are built for the dataloader, personalized models, and tokenizer.

### 3.2.3. Codes to Run
These codes are used for pre-training and fine-tuning.

[training.py](src/training.py) is used for pre-training stage. You can run like this:

```bash
python training.py --dataset 'dataset_name' --lambda_V 1
```
OR
```bash
accelerate launch training.py --dataset 'dataset_name' --lambda_V 1
```

[finetuning.py](src/finetuning.py) is used for fine-tuning stage. You can run like this:

```bash
python finetuning.py --dataset 'dataset_name' --lambda_V 1
```
OR
```bash
accelerate launch finetuning.py --dataset 'dataset_name' --lambda_V 1
```

**Be careful!!!** 
You may need to change the path based on your own.

And, you will need a folder to store the model. It should have a structure like this.

```bash
/'dataset_name'
  /collaborate
  /content
  /rec
```

If you have any questions, please feel free to drop me an e-mail.
