# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import LinearSVC, LinearSVR, SVR
from sklearn import model_selection, tree, linear_model, svm, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import torch
# from sklearn.linear_model import LinearRegression

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from mrmr import mrmr_classif, mrmr_regression
from sklearn.feature_selection import SelectFromModel,SelectKBest, f_regression, f_classif, RFE
from genetic_selection import GeneticSelectionCV
from lassonet import LassoNetClassifierCV, LassoNetRegressorCV
from sklearn.datasets import fetch_california_housing

Random_SEED = 1998

RandomSeed = 723
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# input data and useful files
data_folder = './data/original/'
knockoff_folder = './data/knockoff/new/'
data_label_first = ['Amazon', 'cs', ]
# cls_name = ['Carto', 'Amazon', 'Glycation', 'spectf', 'german_credit', 'uci_credit_card', 'spam_base', 'ionosphere',
#             'HumanActivity', 'higgs', 'pima_indian', 'messidor_features', 'wine_red', 'wine_white', 'yeast',
#             'phpDYCOet']
# cls_name = ['Amazon', 'Glycation', 'spectf', 'german_credit', 'uci_credit_card', 'spam_base', 'ionosphere',
#             'HumanActivity', 'higgs', 'pima_indian', 'messidor_features', 'wine_red', 'wine_white', 'yeast',
#             'phpDYCOet']
cls_name = []
# reg_name = ['housing_boston', 'airfoil', 'openml_618', 'openml_589', 'openml_616', 'openml_607', 'openml_620',
#             'openml_637', 'openml_586']
reg_name = ['california housing']

# name_list = cls_name + reg_name
name_list = ['aaa']


for _ in name_list:
    # choose knockoff type
    # knockoff_type = 'metro'
    # knockoff_type = 'Gaussian'
    #
    # # input data
    # dataset_all = pd.read_csv(knockoff_folder + knockoff_type + '_' + dataset_name + '.csv', header=0)
    # new_column_names = list(range(len(dataset_all.columns)))
    # dataset_all.columns = new_column_names
    # print(dataset_all)
    #
    # # get information
    # r, c = dataset_all.shape
    # n_sample = r
    # n_feature = int((c - 1) / 2)
    # print(n_sample, n_feature)
    #
    # # split data and label
    # if dataset_name in data_label_first:
    #     Y = dataset_all.iloc[:, 0]
    #     X_original = dataset_all.iloc[:, 1:n_feature + 1]
    #     X_knockoff = dataset_all.iloc[:, (n_feature + 1):c]
    # else:
    #     X_original = dataset_all.iloc[:, 0:n_feature]
    #     Y = dataset_all.iloc[:, n_feature]
    #     X_knockoff = dataset_all.iloc[:, (n_feature + 1):c]
    #
    # # create train and validation data
    # X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
    #                                                                   random_state=RandomSeed)

    housing_california = fetch_california_housing()
    dataset_name = 'california housing'
    X_original = pd.DataFrame(housing_california.data)  # data
    Y = pd.DataFrame(housing_california.target)  # label
    n_sample = X_original.shape[0]
    n_feature = X_original.shape[1]
    print(n_sample, n_feature)
    # create train and validation data
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
                                                                      random_state=RandomSeed)
    # X_train = (pd.DataFrame(X_train))
    # X_val = pd.DataFrame(X_val)
    k = int(n_feature / 2)

    # print(type(X_original), X_original.shape)
    # print(type(Y), Y.shape)
    # print(type(k), k)
    # print(X_original.dtypes)
    # print(Y.dtypes)
    # print(X_original)
    # print(Y)
    # X_original = X_original.astype(float)
    # Y = Y.astype(float)

    # define feature selection method
    if dataset_name in reg_name:
        # LASSO
        score_func_lasso = LinearSVR(C=1.0)
        # # mRMR
        # choice_indice_mrmr = torch.LongTensor(mrmr_regression(X_original, Y, K=k, show_progress=False, n_jobs=128))
        choice_indice_mrmr = mrmr_regression(X_original, Y, K=k, show_progress=False, n_jobs=128)
        # GFS
        # estimator_gfs = SVR(kernel="linear")
        # estimator_gfs = LinearRegression()
        estimator_gfs = DecisionTreeRegressor()
        # KBest
        score_func_kbest = f_regression
        # LASSONet
        normalizer_lassonet = preprocessing.Normalizer()
        print('start normalizer_lassonet')
        normalizer_lassonet.fit(X_original)
        print('normalizer_lassonet.fit(X_original)')
        x_lassonrt_reg = normalizer_lassonet.transform(X_original)
        print('x_lassonrt_reg = normalizer_lassonet.transform(X_original)')
        normalizer_lassonet = preprocessing.Normalizer()
        normalizer_lassonet.fit(np.array(Y).reshape(-1, 1))
        print('normalizer_lassonet.fit(np.array(Y).reshape(-1, 1))')
        Y_lassonrt_reg = normalizer_lassonet.transform(np.array(Y).reshape(-1, 1))
        print('Y_lassonrt_reg = normalizer_lassonet.transform(np.array(Y).reshape(-1, 1))')
        selector_lassonet = LassoNetRegressorCV()
        selector_lassonet = selector_lassonet.fit(x_lassonrt_reg, Y_lassonrt_reg)
        print('selector_lassonet = selector_lassonet.fit(x_lassonrt_reg, Y_lassonrt_reg)')
        # RFE
        estimator_rfe = RandomForestRegressor(random_state=0, n_jobs=128)

    elif dataset_name in cls_name:
        # LASSO
        score_func_lasso = LinearSVC(C=1.0, penalty='l1', dual=False)
        # # mRMR
        choice_indice_mrmr = mrmr_classif(X_original, Y, K=k, show_progress=False, n_jobs=128)
        # choice_indice_mrmr = torch.LongTensor(AAA)
        # GFS
        estimator_gfs = DecisionTreeClassifier()
        # KBest
        score_func_kbest = f_classif
        # LASSONet
        normalizer_lassonet = preprocessing.Normalizer()
        normalizer_lassonet.fit(X_original)
        x_lassonrt_cls = normalizer_lassonet.transform(X_original)
        selector_lassonet = LassoNetClassifierCV()  # LassoNetRegressorCV
        selector_lassonet = selector_lassonet.fit(x_lassonrt_cls, Y)
        # RFE
        estimator_rfe = RandomForestRegressor(random_state=0, n_jobs=128)


    # LASSO
    score_func_lasso.fit(X_original, Y)
    model_lasso = SelectFromModel(score_func_lasso, prefit=True, max_features=k)
    # choice_lasso = torch.FloatTensor(model_lasso.get_support())
    choice_lasso = model_lasso.get_support()
    print('choice_lasso = model_lasso.get_support()')
    # # mRMR
    choice_mrmr = choice_indice_mrmr
    # choice_mrmr = torch.zeros(n_feature)
    # choice_mrmr[choice_indice_mrmr-1] = 1
    # GFS
    selector_gfs = GeneticSelectionCV(
        estimator_gfs,
        n_jobs=128,
        max_features=k,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=40, cv=5,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05
    )
    selector_gfs = selector_gfs.fit(X_original, Y)
    # choice_gfs = torch.FloatTensor(selector_gfs.get_support())
    choice_gfs = selector_gfs.get_support()
    print('choice_gfs = selector_gfs.get_support()')
    # KBest
    skb = SelectKBest(score_func=score_func_kbest, k=k)
    skb.fit(X_original, Y)
    # choice_kbest = torch.FloatTensor(skb.get_support())
    choice_kbest = skb.get_support()
    print('choice_kbest = skb.get_support()')
    # LASSONet
    scores_lassonet = selector_lassonet.feature_importances_
    value_lassonet, indice_lassonet = torch.topk(scores_lassonet, k)
    # choice_lassonet = torch.zeros(n_feature)
    # choice_lassonet[indice_lassonet] = 1
    choice_lassonet = indice_lassonet
    print('choice_lassonet = indice_lassonet')
    # RFE
    selector_rfe = RFE(estimator_rfe, n_features_to_select=k, step=1)
    selector_rfe.fit(X_original, Y)
    # choice_rfe = torch.FloatTensor(selector_rfe.get_support())
    choice_rfe = selector_rfe.get_support()
    print('choice_rfe = selector_rfe.get_support()')


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
    for i in range(model_num):


        # LASSO
        choice_numpy_lasso = np.zeros(n_feature)
        choice_numpy_lasso[choice_lasso] = 1.
        selected_indices_lasso = np.where(choice_numpy_lasso == 1)[0]
        if len(selected_indices_lasso) == 0:
            # 从数组的索引中随机选择两个不同的索引
            selected_indices_lasso = np.random.choice(len(choice_numpy_lasso), 2, replace=False)
        X_train_selected_lasso = X_train.iloc[:, selected_indices_lasso]
        X_test_selected_lasso = X_val.iloc[:, selected_indices_lasso]
        model[i].fit(X_train_selected_lasso, Y_train)
        predictions_lasso = model[i].predict(X_test_selected_lasso)
        if dataset_name in reg_name:
            score_lasso = mean_squared_error(Y_val, predictions_lasso)
        elif dataset_name in cls_name:
            score_lasso = accuracy_score(Y_val, predictions_lasso)
        result_all = []
        result_all.append('model_num: ')
        result_all.append(i)
        result_all.append('dataset_name')
        result_all.append(dataset_name)
        result_all.append('score_lasso: ')
        result_all.append(score_lasso)
        file_name = './result/knockoff/baseline/' + dataset_name + '_score_lasso.txt'
        with open(file_name, 'a') as file:
            # 遍历列表中的每个元素
            for item in result_all:
                # 将每个元素写入文件
                file.write("%s\n" % item)

        # mRMR
        choice_numpy_mrmr = np.zeros(n_feature)
        print('choice_mrmr', choice_mrmr)
        choice_numpy_mrmr[np.array(choice_mrmr) - 1] = 1

        # choice_numpy_mrmr[choice_mrmr] = 1
        selected_indices_mrmr = np.where(choice_numpy_mrmr == 1)[0]
        if len(selected_indices_mrmr) == 0:
            # 从数组的索引中随机选择两个不同的索引
            selected_indices_mrmr = np.random.choice(len(choice_numpy_mrmr), 2, replace=False)
        # selected_indices_mrmr = np.where(choice_numpy_mrmr == 1)[0]
        X_train_selected_mrmr = X_train.iloc[:, selected_indices_mrmr]
        X_test_selected_mrmr = X_val.iloc[:, selected_indices_mrmr]
        model[i].fit(X_train_selected_mrmr, Y_train)
        predictions_mrmr = model[i].predict(X_test_selected_mrmr)
        if dataset_name in reg_name:
            score_mrmr = mean_squared_error(Y_val, predictions_mrmr)
        elif dataset_name in cls_name:
            score_mrmr = accuracy_score(Y_val, predictions_mrmr)
        result_all = []
        result_all.append('model_num: ')
        result_all.append(i)
        result_all.append('dataset_name')
        result_all.append(dataset_name)
        result_all.append('score_mrmr: ')
        result_all.append(score_mrmr)
        file_name = './result/knockoff/baseline/' + dataset_name + '_score_mrmr.txt'
        with open(file_name, 'a') as file:
            # 遍历列表中的每个元素
            for item in result_all:
                # 将每个元素写入文件
                file.write("%s\n" % item)

        # GFS
        choice_numpy_gfs = np.zeros(n_feature)
        choice_numpy_gfs[choice_gfs] = 1.
        selected_indices_gfs = np.where(choice_numpy_gfs == 1)[0]
        if len(selected_indices_gfs) == 0:
            # 从数组的索引中随机选择两个不同的索引
            selected_indices_gfs = np.random.choice(len(choice_numpy_gfs), 2, replace=False)
        # selected_indices_gfs = np.where(choice_numpy_gfs == 1)[0]
        X_train_selected_gfs = X_train.iloc[:, selected_indices_gfs]
        X_test_selected_gfs = X_val.iloc[:, selected_indices_gfs]
        model[i].fit(X_train_selected_gfs, Y_train)
        predictions_gfs = model[i].predict(X_test_selected_gfs)
        if dataset_name in reg_name:
            score_gfs = mean_squared_error(Y_val, predictions_gfs)
        elif dataset_name in cls_name:
            score_gfs = accuracy_score(Y_val, predictions_gfs)
        result_all = []
        result_all.append('model_num: ')
        result_all.append(i)
        result_all.append('dataset_name')
        result_all.append(dataset_name)
        result_all.append('score_gfs: ')
        result_all.append(score_gfs)
        file_name = './result/knockoff/baseline/' + dataset_name + '_score_gfs.txt'
        with open(file_name, 'a') as file:
            # 遍历列表中的每个元素
            for item in result_all:
                # 将每个元素写入文件
                file.write("%s\n" % item)

        # KBest
        choice_numpy_kbest = np.zeros(n_feature)
        choice_numpy_kbest[choice_kbest] = 1.
        selected_indices_kbest = np.where(choice_numpy_kbest == 1)[0]
        if len(selected_indices_kbest) == 0:
            # 从数组的索引中随机选择两个不同的索引
            selected_indices_kbest = np.random.choice(len(choice_numpy_kbest), 2, replace=False)
        # selected_indices_kbest = np.where(choice_numpy_kbest == 1)[0]
        X_train_selected_kbest = X_train.iloc[:, selected_indices_kbest]
        X_test_selected_kbest = X_val.iloc[:, selected_indices_kbest]
        model[i].fit(X_train_selected_kbest, Y_train)
        predictions_kbest = model[i].predict(X_test_selected_kbest)
        if dataset_name in reg_name:
            score_kbest = mean_squared_error(Y_val, predictions_kbest)
        elif dataset_name in cls_name:
            score_kbest = accuracy_score(Y_val, predictions_kbest)
        result_all = []
        result_all.append('model_num: ')
        result_all.append(i)
        result_all.append('dataset_name')
        result_all.append(dataset_name)
        result_all.append('score_kbest: ')
        result_all.append(score_kbest)
        file_name = './result/knockoff/baseline/' + dataset_name + '_score_kbest.txt'
        with open(file_name, 'a') as file:
            # 遍历列表中的每个元素
            for item in result_all:
                # 将每个元素写入文件
                file.write("%s\n" % item)

        # LASSONet
        choice_numpy_lassonet = np.zeros(n_feature)
        choice_numpy_lassonet[choice_lassonet] = 1.
        selected_indices_lassonet = np.where(choice_numpy_lassonet == 1)[0]
        if len(selected_indices_lassonet) == 0:
            # 从数组的索引中随机选择两个不同的索引
            selected_indices_lassonet = np.random.choice(len(choice_numpy_lassonet), 2, replace=False)
        # selected_indices_lassonet = np.where(choice_numpy_lassonet == 1)[0]
        X_train_selected_lassonet = X_train.iloc[:, selected_indices_lassonet]
        X_test_selected_lassonet = X_val.iloc[:, selected_indices_lassonet]
        model[i].fit(X_train_selected_lassonet, Y_train)
        predictions_lassonet = model[i].predict(X_test_selected_lassonet)
        if dataset_name in reg_name:
            score_lassonet = mean_squared_error(Y_val, predictions_lassonet)
        elif dataset_name in cls_name:
            score_lassonet = accuracy_score(Y_val, predictions_lassonet)
        result_all = []
        result_all.append('model_num: ')
        result_all.append(i)
        result_all.append('dataset_name')
        result_all.append(dataset_name)
        result_all.append('score_lassonet: ')
        result_all.append(score_lassonet)
        file_name = './result/knockoff/baseline/' + dataset_name + '_score_lassonet.txt'
        with open(file_name, 'a') as file:
            # 遍历列表中的每个元素
            for item in result_all:
                # 将每个元素写入文件
                file.write("%s\n" % item)

        # RFE
        choice_numpy_rfe = np.zeros(n_feature)
        choice_numpy_rfe[choice_rfe] = 1.
        selected_indices_rfe = np.where(choice_numpy_rfe == 1)[0]
        if len(selected_indices_rfe) == 0:
            # 从数组的索引中随机选择两个不同的索引
            selected_indices_rfe = np.random.choice(len(choice_numpy_rfe), 2, replace=False)
        X_train_selected_rfe = X_train.iloc[:, selected_indices_rfe]
        X_test_selected_rfe = X_val.iloc[:, selected_indices_rfe]
        model[i].fit(X_train_selected_rfe, Y_train)
        predictions_rfe = model[i].predict(X_test_selected_rfe)
        if dataset_name in reg_name:
            score_rfe = mean_squared_error(Y_val, predictions_rfe)
        elif dataset_name in cls_name:
            score_rfe = accuracy_score(Y_val, predictions_rfe)
        result_all = []
        result_all.append('model_num: ')
        result_all.append(i)
        result_all.append('dataset_name')
        result_all.append(dataset_name)
        result_all.append('score_rfe: ')
        result_all.append(score_rfe)
        file_name = './result/knockoff/baseline/' + dataset_name + '_score_rfe.txt'
        with open(file_name, 'a') as file:
            # 遍历列表中的每个元素
            for item in result_all:
                # 将每个元素写入文件
                file.write("%s\n" % item)
# print(result_all)
# with open('./result/knockoff/baseline/my_list.txt', 'w') as file:
#     # 遍历列表中的每个元素
#     for item in result_all:
#         # 将每个元素写入文件
#         file.write("%s\n" % item)
