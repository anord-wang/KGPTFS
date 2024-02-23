import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from Laplacian_Score import*


def LaplacianScore_matlab(input_X, X_W):
    nSmp, nFea = input_X.shape()
    if X_W.shape[0] != nSmp:
        print('X_W size is wrong')
    D = sum(X_W,2)
    L = X_W
    tmp1 = np.dot(np.atleast_2d(D),input_X)
    # D = sparse(1:nSmp, 1: nSmp, D, nSmp, nSmp);
    DPrime = sum((input_X.conj().T*D).conj().T * input_X) - tmp1 * tmp1 / sum(np.diag(D))
    LPrime = sum((input_X.conj().T*L).conj().T * input_X) - tmp1 * tmp1 / sum(np.diag(D))
    for i in range(DPrime.shape[0]):
        if DPrime[i][1]<1e-12:
            DPrime[i][1] = 10000
    output_Y = LPrime / DPrime
    output_Y = output_Y.conj().T
    return output_Y



data_folder = './data/'
# dataset = pd.read_csv(data_folder + 'train_Carto.csv')
# dataset = pd.read_csv(data_folder + 'phpDYCOet.csv')
dataset = pd.read_csv(data_folder + 'train_Amazon.csv')
# dataset = pd.read_csv(data_folder + 'train_cs.csv')
# dataset = pd.read_csv(data_folder + 'Glycation.csv')

#
# dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

# rem = ['Id']
# dataset.drop(rem,axis=1,inplace=True)

r, c = dataset.shape
array = dataset.values

X = dataset.iloc[:, 1:c]
Y = dataset.iloc[:, 0]

# print(X)
# print(Y)

#
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)


# Forder = loadmat("IGorderCarto.mat")
# Forder = loadmat("IGorderPHP.mat")
Forder = loadmat("IGorderAmazon.mat")
# Forder = loadmat("IGorderCS.mat")
# Forder = loadmat("IGorderGlycation.mat")
order = Forder["Forder"].squeeze(0) - 1

# print(X_train.shape)
X_train = X_train.iloc[:, order]
X_val_no_order = X_val
X_val = X_val.iloc[:, order]

N_feature = X_train.shape[1]  # feature number
N_sample = X_train.shape[0]  # feature length,i.e., sample number

#
Fstate = np.random.randint(2, size=N_feature)
while sum(Fstate) < 3:
    Fstate = np.random.randint(2, size=N_feature)

X_val_selected = X_train.iloc[:, Fstate == 1]

print(X_train.shape)
print(X_val_selected.shape)
print(X_val_no_order.shape)


# L_s_1 = LaplacianScore(X_val, neighbour_size=4, t_param=2)
# print(L_s_1)
L_s_2 = LaplacianScore(X_val_selected, neighbour_size=4, t_param=2)
print(L_s_2)
L_s_3 = LaplacianScore(X_val_no_order, neighbour_size=4, t_param=2)
print(L_s_3)
