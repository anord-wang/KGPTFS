# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
import Representation_learning
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection, tree, linear_model, svm, neighbors
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.datasets import fetch_california_housing
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

# from xgboost.sklearn import XGBClassifier
# import matplotlib.pyplot as plt

RandomSeed = 723

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # input data and useful files
Random_SEED = 1998
data_folder = './data/original/'
knockoff_folder = './data/knockoff/new/'
knockoff_label_folder = './data/knockoff/knockoff_result/label/'
data_label_first = ['Amazon', 'cs', ]
mat_file_folder = './data/search_order/'

cls_name = ['AP_Omentum_Ovary', 'german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
            'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
            'wine_red', 'wine_white', 'yeast', 'HumanActivity', 'cs', 'phpDYCOet']
# cls_name = ['german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
#             'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
#             'wine_red', 'wine_white', 'yeast', 'HumanActivity', 'cs', 'phpDYCOet']
reg_name = ['airfoil', 'bike_share', 'blogData', 'housing_boston', 'openml_586', 'openml_589', 'openml_607',
            'openml_616', 'openml_618', 'openml_620', 'openml_637']
# cls_name = ['german_credit']
# reg_name = []
name_list = cls_name + reg_name
order_list = ['cs', 'phpDYCOet']
for dataset_name in name_list:
    # choose knockoff type
    # knockoff_type = 'metro'
    knockoff_type = 'Gaussian'

    # choose measurement method
    measure_type = 'Euclidean'

    # choose threshold type
    threshold_type = 'mean'
    # threshold_type = 'median'

    # input data
    dataset_all = pd.read_csv(knockoff_folder + knockoff_type + '_' + dataset_name + '.csv')

    # input knockoff label
    # knockoff_label = np.load(knockoff_label_folder + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy')

    # get information
    r, c = dataset_all.shape
    n_sample = r
    n_feature = int((c - 1) / 2)
    print(n_sample, n_feature)

    # split data and label
    if dataset_name in data_label_first:
        Y = dataset_all.iloc[:, 0]
        X_original = dataset_all.iloc[:, 1:n_feature + 1]
        X_knockoff = dataset_all.iloc[:, (n_feature + 1):c]
    else:
        X_original = dataset_all.iloc[:, 0:n_feature]
        Y = dataset_all.iloc[:, n_feature]
        X_knockoff = dataset_all.iloc[:, (n_feature + 1):c]

    # create train and validation data
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
                                                                      random_state=Random_SEED)

    # initial model
    if dataset_name in reg_name:
        model_1 = ensemble.AdaBoostRegressor(n_estimators=50)  # 用50个决策树
        model_2 = tree.DecisionTreeRegressor()
        model_3 = linear_model.LinearRegression()
        model_4 = svm.SVR()
        model_5 = neighbors.KNeighborsRegressor()
        model_6 = ensemble.RandomForestRegressor(n_estimators=20)  # 用20个决策树
        model_7 = ensemble.GradientBoostingRegressor(n_estimators=100)  # 用100个决策树
        model_8 = ensemble.BaggingRegressor()
        model_9 = tree.ExtraTreeRegressor()
        model = (model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9)
        model_num = 9
    elif dataset_name in cls_name:
        # model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
        model_1 = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
        model_2 = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)
        model_3 = GaussianNB()
        model_4 = svm.SVC()
        model_5 = MLPClassifier()
        model_6 = ensemble.BaggingClassifier()
        model_7 = ensemble.AdaBoostClassifier()
        model_8 = ensemble.GradientBoostingClassifier()
        model = (model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8)
        model_num = 8
    # load feature order
    if dataset_name in order_list:
        mat_file_name = mat_file_folder + 'IGorder' + dataset_name + '.mat'
        Forder = loadmat(mat_file_name)
        order = Forder["Forder"].squeeze(0) - 1
        X_train = X_train.iloc[:, order]
        X_val = X_val.iloc[:, order]

    # ======================================================================================================================

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    np.random.seed(Random_SEED)
    torch.manual_seed(Random_SEED)  # reproducible

    result_all = [[] for i in range(model_num)]

    if dataset_name in reg_name:

        for model_index in range(model_num):
            model[model_index].fit(X_train, Y_train)
            pred_y = model[model_index].predict(X_val)
            mse = mean_squared_error(Y_val, pred_y)
            result_all[model_index].append(mse)

            # sum = 0
            # for i in range(10):
            #     L1 = random.sample(range(0, n_feature), int(n_feature / 2))
            #     Fstate = np.zeros(8)
            #     Fstate[[index for index in L1]] = 1
            #     # print(Fstate)
            #     X_train_random = X_train[:, Fstate == 1]
            #     X_val_random = X_val[:, Fstate == 1]
            #     model[model_index].fit(X_train_random, Y_train)
            #     pred_y = model[model_index].predict(X_val_random)
            #     mse = mean_squared_error(Y_val, pred_y)
            #     # accuracy = model.score(X_val_random, Y_val)
            #     # print(i, mse)
            #     sum = sum + mse
            # result_all[model_index].append(sum / 10)



    elif dataset_name in cls_name:

        for model_index in range(model_num):
            model[model_index].fit(X_train, Y_train)
            accuracy = model[model_index].score(X_val, Y_val)
            Y_pred = model[model_index].predict(X_val)
            # macroF1 = f1_score(Y_val, Y_pred, average='macro')
            # precision = precision_score(Y_val, Y_pred, average='macro')
            # recall = recall_score(Y_val, Y_pred, average='macro')
            result_all[model_index].append(accuracy)

            # sum = 0
            # for i in range(10):
            #     L1 = random.sample(range(0, n_feature), int(n_feature / 2))
            #     Fstate = np.zeros(8)
            #     Fstate[[index for index in L1]] = 1
            #     # print(Fstate)
            #     X_train_random = X_train[:, Fstate == 1]
            #     X_val_random = X_val[:, Fstate == 1]
            #     model[model_index].fit(X_train_random, Y_train)
            #     accuracy = model[model_index].score(X_val_random, Y_val)
            #     Y_pred = model[model_index].predict(X_val_random)
            #     # macroF1 = f1_score(Y_val, Y_pred, average='macro')
            #     # precision = precision_score(Y_val, Y_pred, average='macro')
            #     # recall = recall_score(Y_val, Y_pred, average='macro')
            #     sum = sum + accuracy
            # result_all[model_index].append(sum / 10)
            # # result_all[model_index].append([sum / 10, precision, recall, macroF1, Fstate])

    output_all = [[] for i in range(model_num)]
    reward_output_all = [[] for i in range(model_num)]
    name_all = [[] for i in range(model_num)]

    for model_index in range(model_num):
        result = result_all[model_index]
        output = output_all[model_index]
        name = name_all[model_index]

        name.append("result types")
        output.append('many many models, No ' + str(model_index + 1) + 'ALL feature NO Feature Selection')

        max_accuracy = 0
        min_mse = 100000000
        optimal_set = []
        if dataset_name in reg_name:
            name.append('MSE')
            output.append(result[0])

        elif dataset_name in cls_name:
            name.append('ACC')
            output.append(result[0])

        out_1 = dict(zip(name, output))
        out_1 = pd.DataFrame([out_1])
        result_folder = './result/knockoff/'
        out_1.to_csv(result_folder + dataset_name + '_test.csv', mode='a')
