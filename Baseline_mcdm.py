# filter methods
import warnings

warnings.filterwarnings("ignore")
import mcdm
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import safe_sqr
from operator import attrgetter

from xgboost import XGBClassifier, XGBRegressor

# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, LinearSVR, SVR
from sklearn import model_selection, tree, linear_model, svm, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import torch
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, f_classif, RFE
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
# cls_name = ['wine_red', 'wine_white', 'yeast', 'phpDYCOet']
cls_name = []
# reg_name = ['housing_boston', 'airfoil', 'openml_618', 'openml_589', 'openml_616', 'openml_607', 'openml_620',
#             'openml_637', 'openml_586']
reg_name = ['california housing']
# name_list = cls_name + reg_name
# name_list = reg_name + cls_name
name_list = ['aaa']

def get_feature_importances(estimator, getter, transform_func=None, norm_order=1):
    """
    Retrieve and aggregate (ndim > 1)  the feature importances
    from an estimator. Also optionally applies transformation.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator from which we want to get the feature
        importances.

    getter : "auto", str or callable
        An attribute or a callable to get the feature importance. If `"auto"`,
        `estimator` is expected to expose `coef_` or `feature_importances`.

    transform_func : {"norm", "square"}, default=None
        The transform to apply to the feature importances. By default (`None`)
        no transformation is applied.

    norm_order : int, default=1
        The norm order to apply when `transform_func="norm"`. Only applied
        when `importances.ndim > 1`.

    Returns
    -------
    importances : ndarray of shape (n_features,)
        The features importances, optionally transformed.
    """
    if isinstance(getter, str):
        if getter == "auto":
            if hasattr(estimator, "coef_"):
                getter = attrgetter("coef_")
            elif hasattr(estimator, "feature_importances_"):
                getter = attrgetter("feature_importances_")
            else:
                raise ValueError(
                    "when `importance_getter=='auto'`, the underlying "
                    f"estimator {estimator.__class__.__name__} should have "
                    "`coef_` or `feature_importances_` attribute. Either "
                    "pass a fitted estimator to feature selector or call fit "
                    "before calling transform."
                )
        else:
            getter = attrgetter(getter)
    elif not callable(getter):
        raise ValueError("`importance_getter` has to be a string or `callable`")

    importances = getter(estimator)

    if transform_func is None:
        return importances
    elif transform_func == "norm":
        if importances.ndim == 1:
            importances = np.abs(importances)
        else:
            importances = np.linalg.norm(importances, axis=0, ord=norm_order)
    elif transform_func == "square":
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)
    else:
        raise ValueError(
            "Valid values for `transform_func` are "
            + "None, 'norm' and 'square'. Those two "
            + "transformation are only supported now"
        )

    return importances


def Kbest(X, y, task_type):
    if task_type == 'reg':
        score_func = f_regression
    else:
        score_func = f_classif
    skb = SelectKBest(score_func=score_func)
    score_func_ret = skb.score_func(X, y)
    if isinstance(score_func_ret, (list, tuple)):
        scores_, pvalues_ = score_func_ret
        # pvalues_ = np.asarray(pvalues_)
    else:
        scores_ = score_func_ret
        # pvalues_ = None
    scores_ = np.asarray(scores_)
    return scores_


def LASSO(X, y, task_type):
    if task_type == 'reg':
        score_func = LinearSVR(C=1.0)
    else:
        score_func = LinearSVC(C=1.0, penalty='l1', dual=False)
    model = SelectFromModel(score_func)
    model.fit(X, y)
    return get_feature_importances(model.estimator_, getter='auto')


def Rfe(X, y, task_type):
    # k = X.shape[1] / 2
    if task_type == 'reg':
        # estimator = SVR(kernel="linear")
        estimator = RandomForestRegressor(random_state=0, n_jobs=128)
    else:
        estimator = RandomForestClassifier(random_state=0, n_jobs=128)
        # estimator = RandomForestClassifier(max_depth=7, random_state=0, n_jobs=128)
    selector = RFE(estimator, n_features_to_select=0.5, step=1)
    selector.fit(X, y)
    choice = selector.get_support(True)
    imp = get_feature_importances(selector.estimator_, getter='auto')
    score = torch.zeros(X.shape[1])
    for ind, i in enumerate(choice):
        if i == 0:
            continue
        else:
            score[i] = imp[ind]
    return score


funcs = [LASSO, Kbest, Rfe]


def rest(X, y, task_type):
    if task_type == 'reg':
        return [i(X, y, task_type) for i in funcs]
    imps = []
    dep = 12
    for method in ['RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'DT']:
        if method == 'RF':
            if task_type == 'cls':
                model = RandomForestClassifier(random_state=0, n_jobs=128)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
            else:
                model = RandomForestRegressor(max_depth=dep, random_state=0, n_jobs=128)
        elif method == 'XGB':
            if task_type == 'cls':
                model = XGBClassifier(eval_metric='logloss', n_jobs=128)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=2, n_jobs=128)
                # continue
                model = XGBRegressor(eval_metric='logloss', n_jobs=128)
        elif method == 'SVM':
            if task_type == 'cls':
                model = LinearSVC()
            elif task_type == 'mcls':
                model = LinearSVC()
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=3, n_jobs=128)
                # continue
                model = LinearSVR()
        elif method == 'Ridge':
            if task_type == 'cls':
                model = RidgeClassifier()
            elif task_type == 'mcls':
                model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=5, n_jobs=128)
                # continue
                model = Ridge()
        elif method == 'LASSO':
            if task_type == 'cls':
                model = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=128)
            elif task_type == 'mcls':
                model = OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear'), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=8, n_jobs=128)
                model = DecisionTreeRegressor(max_depth=7, random_state=1)
                # continue
                # model = Lasso()
        else:  # dt
            if task_type == 'cls':
                model = DecisionTreeClassifier()
            elif task_type == 'mcls':
                model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
            else:
                # model = RandomForestRegressor(max_depth=dep, random_state=12, n_jobs=128)
                # continue
                model = DecisionTreeRegressor(max_depth=dep)
        selector = SelectFromModel(model)
        selector.fit(X, y)
        if task_type == 'mcls':
            if method != 'SVM':
                overall_imp = []
                for i in selector.estimator_.estimators_:
                    overall_imp.append(get_feature_importances(i, getter='auto'))
                imps.append(np.concatenate([i.reshape(-1, 1) for i in overall_imp], 1).mean(1))
            else:
                overall_imp = get_feature_importances(selector.estimator_, getter='auto')
                imps.append(overall_imp.mean(0))
        else:
            score = get_feature_importances(selector.estimator_, getter='auto')
            # if task_type == 'reg':
            #     score_ = torch.LongTensor(score.argsort())
            #     choice_index = score_[:k]
            #     choice = torch.zeros(score_.shape[0])
            #     choice[choice_index] = 1
            #     test_result = fe.report_performance(choice, flag='test', store=False, rp=False)
            #     print(f'{method}', choice_index, test_result)
            imps.append(score)
    return imps


for _ in name_list:
    # # choose knockoff type
    # # knockoff_type = 'metro'
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
    # print('Y', Y)

    # while True:
    #     min_value = Y.min()
    #     print('min_value', min_value)
    #     if min_value != 0:
    #         Y = Y - 1
    #         print('Y', Y)
    #     else:
    #         break
    # create train and validation data
    # X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
    #                                                                   random_state=RandomSeed)

    housing_california = fetch_california_housing()
    dataset_name = 'california housing'
    X_original = housing_california.data  # data
    Y = housing_california.target  # label
    n_sample = X_original.shape[0]
    n_feature = X_original.shape[1]
    print(n_sample, n_feature)
    # create train and validation data
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
                                                                      random_state=RandomSeed)

    k = int(n_feature / 2)

    if dataset_name in reg_name:
        task_type = 'reg'
    elif dataset_name in cls_name:
        task_type = 'cls'


    accumulated = rest(X_original, Y, task_type)
    norm_importance = []
    for labels in accumulated:
        labels = labels.reshape(-1)
        min_val = min(labels)
        max_val = max(labels)
        train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
        norm_importance.append(train_encoder_target)
    print(len(norm_importance))
    print([len(x) for x in norm_importance])
    norm_importance_truncated = [arr[:len(norm_importance[0])] for arr in norm_importance]
    importances = torch.FloatTensor(norm_importance_truncated).reshape(len(norm_importance[0]), len(norm_importance))
    # importances = torch.FloatTensor(norm_importance).reshape(len(norm_importance[0]), len(norm_importance))
    order = importances.argsort(descending=True)
    score = torch.zeros_like(order, dtype=torch.float)
    for index, i in enumerate(order):
        for j, pos in zip(range(order.shape[1]), i):
            score[index, pos] = (order.shape[1] - j - 1 + 0.) / order.shape[1]
    alt_name = [str(i) for i in range(X_original.shape[1])]
    # print(importances)
    if task_type == 'reg':
        rank = mcdm.rank(importances, s_method="TOPSIS", n_method="Linear1",
                         c_method="AbsPearson",
                         w_method="VIC", alt_names=alt_name)
    else:
        rank = mcdm.rank(importances, s_method="TOPSIS", n_method="Linear1",
                         c_method="AbsPearson",
                         w_method="VIC", alt_names=alt_name)
    print('aggre', [int(i) for i, j in rank])
    selected = rank[:k]
    choice_index = torch.LongTensor([int(i) for i, score in selected])
    # info(f'current selection is {choice}')
    choice = torch.zeros(n_feature)
    choice[choice_index] = 1

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
        # X_train_selected_lasso = X_train[:, choice]
        choice_numpy = choice.numpy()

        # 获取 True 值的索引
        selected_indices = np.where(choice_numpy == 1)[0]
        X_train_selected_lasso = X_train[:, selected_indices]
        X_test_selected_lasso = X_val[:, selected_indices]
        model[i].fit(X_train_selected_lasso, Y_train)
        predictions_lasso = model[i].predict(X_test_selected_lasso)
        if dataset_name in reg_name:
            score_mcdm = mean_squared_error(Y_val, predictions_lasso)
        elif dataset_name in cls_name:
            score_mcdm = accuracy_score(Y_val, predictions_lasso)

        result_all = []
        result_all.append('model_num: ')
        result_all.append(model_num)
        result_all.append('dataset_name')
        result_all.append(dataset_name)
        result_all.append('score_mcdm: ')
        result_all.append(score_mcdm)
        file_name = './result/knockoff/baseline/' + dataset_name + '_' + str(model_num) + '_score_mcdm.txt'
        with open(file_name, 'a') as file:
            # 遍历列表中的每个元素
            for item in result_all:
                # 将每个元素写入文件
                file.write("%s\n" % item)
