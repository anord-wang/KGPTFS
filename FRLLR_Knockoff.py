# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
import Representation_learning
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection,tree,linear_model,svm,neighbors
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
#

class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS, knockoff_label=None, model_path=None):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        state_dict = torch.load(model_path)
        self.eval_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

        self.learn_step_counter = 0
        self.EPSILON = EPSILON
        self.knockoff_EPSILON = knockoff_EPSILON
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON_STEP_SIZE = EPSILON_STEP_SIZE
        self.knockoff_label = knockoff_label

        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        feature_list = x[N_STATES - N_feature:N_STATES]
        feature_index = [index for index, e in enumerate(feature_list) if e == 1]
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        random_1 = np.random.uniform()
        random_2 = np.random.uniform()
        if random_1 < self.EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            action_value = action_value.tolist()[0]
        else:
            if random_2 < self.knockoff_EPSILON:
                action = np.random.randint(0, N_ACTIONS)
                print('random:', action)
            else:
                print('feature_index:', feature_index)
                print(self.knockoff_label[feature_index])
                action = self.knockoff_label[feature_index]
                print('knockoff random:', action)
            if self.EPSILON <= self.EPSILON_MAX:
                self.EPSILON = self.EPSILON + self.EPSILON_STEP_SIZE
            action_value = [int(1 - action), action]

        return action, action_value

    def store_transition(self, s, a, r, s_):
        # transition = np.hstack((s, [a, r], s_))
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY  # If full, restart from the beginning
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
# cls_name = ['AP_Omentum_Ovary', 'german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
#             'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
#             'wine_red', 'wine_white', 'yeast', 'HumanActivity', 'cs', 'phpDYCOet']
# cls_name = ['german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
#             'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
#             'wine_red', 'wine_white', 'yeast', 'HumanActivity', 'cs', 'phpDYCOet']
cls_name = []
# reg_name = ['airfoil', 'bike_share', 'blogData', 'housing_boston', 'openml_586', 'openml_589', 'openml_607',
#             'openml_616', 'openml_618', 'openml_620', 'openml_637']
reg_name = ['openml_586', 'openml_589', 'openml_607', 'openml_616', 'openml_618', 'openml_620', 'openml_637']
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
    knockoff_label = np.load(knockoff_label_folder + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy')

    # get information
    r, c = dataset_all.shape
    n_sample = r
    n_feature = int((c - 1) / 2)
    print(n_sample, n_feature)

    # split data and label
    if dataset_name in data_label_first:
        Y = dataset_all.iloc[:, 0]
        X_original = dataset_all.iloc[:, 1:n_feature+1]
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
        model_6 = ensemble.RandomForestRegressor(n_estimators=20)#用20个决策树
        model_7 = ensemble.GradientBoostingRegressor(n_estimators=100)#用100个决策树
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

    # DQN
    N_feature = X_train.shape[1]  # feature number
    N_sample = X_train.shape[0]  # feature length,i.e., sample number
    BATCH_SIZE = 16
    LR = 0.01
    EPSILON = 0.9  # 90%的概率使用DQN网络选择动作
    EPSILON_MAX = 1.0  # 最终不使用随机选择
    EPSILON_STEP_SIZE = 0.001
    knockoff_EPSILON = 0.5  # 在随机选择中使用knockoff标签的比例
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 100  # After how much time you refresh target network
    MEMORY_CAPACITY = 400  # The size of experience replay buffer
    EXPLORE_STEPS = max(N_feature * 10, 1000)  # How many exploration steps you'd like, should be larger than MEMORY_CAPACITY20
    VAL_STEPS = N_feature
    N_ACTIONS = 2
    N_STATES = 480 + N_feature
    knockoff_p = 0.7

    np.random.seed(Random_SEED)
    torch.manual_seed(Random_SEED)  # reproducible


    model_save_folder = './data/knockoff/knockoff_result/model/'
    current_path = model_save_folder + 'Gaussian_' + dataset_name + '_' + measure_type + '_' + threshold_type + '_newnet.pth'
    dqn = DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS, knockoff_label=knockoff_label, model_path=current_path)

    # # The element in the result list consists two parts,
    # # i.e., accuracy and the action list (action 1 means selecting corresponding feature, 0 means deselection).
    #
    L1 = random.sample(range(0, N_feature), 2)
    Fstate = np.zeros(N_feature)
    Fstate[[index for index in L1]] = 1

    # Fstate = np.random.randint(2, size=N_feature)
    # while sum(Fstate) < 2:
    #     Fstate = np.random.randint(2, size=N_feature)

    # Fstate = np.zeros(N_feature)
    # Fstate[0] = 1
    X_selected = X_train.iloc[:, Fstate == 1]
    X_array = np.array(X_selected)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_array = min_max_scaler.fit_transform(X_array)
    X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
    s = Representation_learning.representation_training(X_tensor)
    s = s.detach().numpy().reshape(-1)
    position = np.arange(1, N_feature + 1)
    onehot_encoded = OneHotEncoder(sparse=False).fit_transform(position.reshape(-1, 1))
    s = np.append(s, onehot_encoded[0])
    # s = np.append(s,1)
    knockoff_select_list = np.zeros(N_feature)

    # result = []
    result_all = [[] for i in range(model_num)]

    T = N_feature
    # dqn.EPSILON_STEP_SIZE = (dqn.EPSILON_MAX - dqn.EPSILON)/((EXPLORE_STEPS-1)*T)
    for i in range(EXPLORE_STEPS):
        t = i % T
        Faction, Factionvalue = dqn.choose_action(s)
        Fstate[t] = Faction
        if sum(Fstate) < 1:
            Faction = 1
            Fstate[t] = Faction

        X_selected = X_train.iloc[:, Fstate == 1]
        X_array = np.array(X_selected)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        X_array = min_max_scaler.fit_transform(X_array)
        X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
        s_ = Representation_learning.representation_training(X_tensor)
        s_ = s_.detach().numpy().reshape(-1)
        if t == T - 1:
            s_ = np.append(s_, onehot_encoded[0])
            # s_ = np.append(s_, 1)
        else:
            s_ = np.append(s_, onehot_encoded[t + 1])
            # s_ = np.append(s_, t + 2)

        corr = X_val.corr().abs()
        ave_corr = (corr.iloc[:, t].sum()) / (X_val.shape[1])

        # ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
        # r = (accuracy - ave_corr)
        # r = accuracy

        feature_list = s[N_STATES - N_feature:N_STATES]
        feature_index = [index for index, e in enumerate(feature_list) if e == 1]
        knockoff_para = knockoff_label[feature_index]
        knockoff_likehood = abs(np.float64(Factionvalue[1]))
        knockoff_reward = (knockoff_p ** int(knockoff_select_list[feature_index])) * int(
            1 - knockoff_para) * Faction * knockoff_likehood
        knockoff_select_list[feature_index] = knockoff_select_list[feature_index] + Faction

        # ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
        # r = (accuracy - knockoff_reward - ave_corr)
        r = (1 + knockoff_reward - ave_corr)
        # r = knockoff_reward
        # r = accuracy

        dqn.store_transition(s, Faction, r, s_)

        # 每100步计算一次acc进行保存
        if (i != 0) and (i % VAL_STEPS == 0):
            if dataset_name in reg_name:

                for model_index in range(model_num):
                    model[model_index].fit(X_train.iloc[:, Fstate == 1], Y_train)
                    pred_y = model[model_index].predict(X_val.iloc[:, Fstate == 1])
                    mse = mean_squared_error(Y_val, pred_y)
                    result_all[model_index].append([mse, Fstate, r])

                # model.fit(X_train.iloc[:, Fstate == 1], Y_train)
                # pred_y = model.predict(X_val.iloc[:, Fstate == 1])
                # mse = mean_squared_error(Y_val, pred_y)
                # r = 1 - compareLodge_loss.detach().numpy() - ave_corr
                # result.append([mse, Fstate, r])

            elif dataset_name in cls_name:

                for model_index in range(model_num):
                    model[model_index].fit(X_train.iloc[:, Fstate == 1], Y_train)
                    accuracy = model[model_index].score(X_val.iloc[:, Fstate == 1], Y_val)
                    Y_pred = model[model_index].predict(X_val.iloc[:, Fstate == 1])
                    macroF1 = f1_score(Y_val, Y_pred, average='macro')
                    precision = precision_score(Y_val, Y_pred, average='macro')
                    recall = recall_score(Y_val, Y_pred, average='macro')
                    print(r, accuracy)
                    result_all[model_index].append([accuracy, precision, recall, macroF1, Fstate, r])

                # model.fit(X_train.iloc[:, Fstate == 1], Y_train)
                # accuracy = model.score(X_val.iloc[:, Fstate == 1], Y_val)
                # Y_pred = model.predict(X_val.iloc[:, Fstate == 1])
                # macroF1 = f1_score(Y_val, Y_pred, average='macro')
                # precision = precision_score(Y_val, Y_pred, average='macro')
                # recall = recall_score(Y_val, Y_pred, average='macro')
                # r = accuracy - ave_corr
                # result.append([accuracy, precision, recall, macroF1, Fstate, r])

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        s = s_

    # output = []
    # reward_output = []
    # name = []
    output_all = [[] for i in range(model_num)]
    reward_output_all = [[] for i in range(model_num)]
    name_all = [[] for i in range(model_num)]

    for model_index in range(model_num):
        result = result_all[model_index]
        output = output_all[model_index]
        reward_output = reward_output_all[model_index]
        name = name_all[model_index]

        name.append("result types")
        output.append('Only Knockoff compare loss reward, many many models, No '+str(model_index+1))
        reward_output.append('Only Knockoff compare loss reward, many many models, No '+str(model_index+1))

        max_accuracy = 0
        min_mse = 100000000
        optimal_set = []
        for i in range(len(result)):
            if dataset_name in reg_name:
                name.append("MSE of the {}-th explore step".format(i * VAL_STEPS))
                output.append(result[i][0])
                reward_output.append(result[i][2])
                if result[i][0] < min_mse:
                    min_mse = result[i][0]
                    optimal_set = result[i][1]
            elif dataset_name in cls_name:
                name.append("ACC of the {}-th explore step".format(i * VAL_STEPS))
                output.append(result[i][0])
                reward_output.append(result[i][5])
                if result[i][0] > max_accuracy:
                    max_accuracy = result[i][0]
                    optimal_set = result[i][4]
                    Mmacro_f1 = result[i][3]
                    Mprecision = result[i][1]
                    Mrecall = result[i][2]

        if dataset_name in reg_name:
            print("The min_mse is: {}, the optimal selection for each feature is:{}".format(min_mse, optimal_set))
            name.append("feature subset")
            output.append(optimal_set)
            name.append("min_mse")
            output.append(min_mse)
        elif dataset_name in cls_name:
            print("The maximum accuracy is: {}, the optimal selection for each feature is:{}".format(max_accuracy, optimal_set))
            name.append("feature subset")
            output.append(optimal_set)
            name.append("max_accuracy")
            output.append(max_accuracy)
            name.append("precision")
            output.append(Mprecision)
            name.append("recall_RF")
            output.append(Mrecall)
            name.append("macro_f1")
            output.append(Mmacro_f1)


        out_1 = dict(zip(name, output))
        out_2 = dict(zip(name, reward_output))
        out_1 = pd.DataFrame([out_1])
        out_2 = pd.DataFrame([out_2])
        result_folder = './result/knockoff/'
        out_1.to_csv(result_folder + 'new_' + dataset_name + '_test.csv', mode='a')
        out_2.to_csv(result_folder + 'new_' + dataset_name + '_reward_test.csv', mode='a')