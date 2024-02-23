# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
import Representation_learning
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.datasets import fetch_california_housing
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

# from xgboost.sklearn import XGBClassifier
# import matplotlib.pyplot as plt

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # input data and useful files
Random_SEED = 723
# knockoff_folder = './data/knockoff/'
# data_label_first = ['Amazon', 'cs', ]
# mat_file_folder = './data/search_order/'
#
# # choose dataset
# # dataset_name = 'shortCarto'
# # dataset_name = 'Carto'
# # dataset_name = 'phpDYCOet'
# # dataset_name = 'Amazon'
# # dataset_name = 'cs'
# dataset_name = 'Glycation'
#
# # choose knockoff type
# knockoff_type = 'metro'
# # knockoff_type = 'Gaussian'
#
# # input data
# dataset_all = pd.read_csv(knockoff_folder + knockoff_type + '_' + dataset_name + '.csv')
#
# # get information
# r, c = dataset_all.shape
# n_sample = r
# n_feature = int((c - 1) / 2)
#
# # split data and label
# if dataset_name in data_label_first:
#     Y = dataset_all.iloc[:, 0]
#     X_original = dataset_all.iloc[:, 1:n_feature+1]
#     X_knockoff = dataset_all.iloc[:, (n_feature + 1):c]
# else:
#     X_original = dataset_all.iloc[:, 0:n_feature]
#     Y = dataset_all.iloc[:, n_feature]
#     X_knockoff = dataset_all.iloc[:, (n_feature + 1):c]
#
# # create train and validation data
# X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
#                                                                   random_state=Random_SEED)
#
# # initial model
# model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
# # model =DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)
#
# # load feature order
# mat_file_name = mat_file_folder + 'IGorder' + dataset_name + '.mat'
# Forder = loadmat(mat_file_name)
# order = Forder["Forder"].squeeze(0) - 1
#
# # put features in order
# X_train = X_train.iloc[:, order]
# X_val = X_val.iloc[:, order]
# # X_original = X_original.iloc[:, order]
# # X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
# #                                                                   random_state=Random_SEED)
#
# sum = 0
# for i in range(10):
#     L1 = random.sample(range(0, 402), 201)
#     Fstate = np.zeros(402)
#     Fstate[[index for index in L1]] = 1
#     # print(Fstate)
#     X_train_random = X_train.iloc[:, Fstate == 1]
#     X_val_random = X_val.iloc[:, Fstate == 1]
#     model.fit(X_train_random, Y_train)
#     accuracy = model.score(X_val_random, Y_val)
#     print(i, accuracy)
#     sum = sum + accuracy
#
# print(sum/10)

#
# model.fit(X_train, Y_train)
# accuracy = model.score(X_val, Y_val)
# print(accuracy)
# Y_pred = model.predict(X_val)
# macroF1 = f1_score(Y_val, Y_pred, average='macro')
# precision = precision_score(Y_val, Y_pred, average='macro')
# recall = recall_score(Y_val, Y_pred, average='macro')
#
# name = []
# output = []
# name.append("accuracy")
# output.append(accuracy)
# name.append("precision")
# output.append(precision)
# name.append("recall_RF")
# output.append(recall)
# name.append("macro_f1")
# output.append(macroF1)
#
# out_1 = dict(zip(name, output))
# out_1 = pd.DataFrame([out_1])
#
# result_folder = './result/knockoff/'
# out_1.to_csv(result_folder + dataset_name + 'all_feature.csv', mode='a')
#


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# regression task
model = ensemble.AdaBoostRegressor(n_estimators=50)  # 用50个决策树
# model = tree.DecisionTreeRegressor()
# model = linear_model.LinearRegression()
# model = svm.SVR()
# model = neighbors.KNeighborsRegressor()
# model = ensemble.RandomForestRegressor(n_estimators=20)#用20个决策树
# model = ensemble.GradientBoostingRegressor(n_estimators=100)#用100个决策树
# model = BaggingRegressor()
# model = ExtraTreeRegressor()

housing_california = fetch_california_housing()
dataset_name = 'california housing'
LR_X = housing_california.data  # data
LR_y = housing_california.target  # label
n_sample = LR_X.shape[0]
n_feature = LR_X.shape[1]
print(n_sample, n_feature)
# create train and validation data
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(LR_X, LR_y, test_size=0.1, random_state=Random_SEED)

model.fit(X_train, Y_train)
pred_y = model.predict(X_val)
mse = mean_squared_error(Y_val, pred_y)
print('all feature: ', mse)

sum = 0
for i in range(10):
    L1 = random.sample(range(0, 8), 4)
    Fstate = np.zeros(8)
    Fstate[[index for index in L1]] = 1
    # print(Fstate)
    X_train_random = X_train[:, Fstate == 1]
    X_val_random = X_val[:, Fstate == 1]
    model.fit(X_train_random, Y_train)
    pred_y = model.predict(X_val_random)
    mse = mean_squared_error(Y_val, pred_y)
    # accuracy = model.score(X_val_random, Y_val)
    print(i, mse)
    sum = sum + mse

print(sum/10)







