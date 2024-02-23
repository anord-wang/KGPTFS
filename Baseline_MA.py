# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
Random_SEED = 1998
from sklearn.datasets import fetch_california_housing

RandomSeed = 723
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# input data and useful files
data_folder = './data/original/'
knockoff_folder = './data/knockoff/new/'
data_label_first = ['Amazon', 'cs', ]
# cls_name = ['Carto', 'Amazon', 'Glycation', 'spectf', 'german_credit', 'uci_credit_card', 'spam_base', 'ionosphere',
#             'HumanActivity', 'higgs', 'pima_indian', 'messidor_features', 'wine_red', 'wine_white', 'yeast',
#             'phpDYCOet']
cls_name = ['higgs', 'pima_indian', 'messidor_features', 'wine_red', 'wine_white', 'yeast',
            'phpDYCOet']
# reg_name = ['housing_boston', 'airfoil', 'openml_618', 'openml_589', 'openml_616', 'openml_607', 'openml_620',
#             'openml_637', 'openml_586']
reg_name = ['california housing']
# name_list = cls_name + reg_name
# name_list = reg_name + cls_name
name_list = ['aaa']

result_all = []

def Feature_GCN(X):
    corr_matrix = pd.DataFrame(X).corr().abs()
    corr_matrix[np.isnan(corr_matrix)] = 0
    corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
    sum_vec = corr_matrix_.sum()

    for i in range(len(corr_matrix_)):
        corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
        corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
    W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(pd.DataFrame(X).values, W.values), axis=1)

    return Feature

class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
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


# for dataset_name in name_list:
for _ in name_list:
    # # choose knockoff type
    # # knockoff_type = 'metro'
    # knockoff_type = 'Gaussian'
    #
    # # input data
    # dataset_all = pd.read_csv(knockoff_folder + knockoff_type + '_' + dataset_name + '.csv')
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
    LR_X = housing_california.data  # data
    LR_y = housing_california.target  # label
    n_sample = LR_X.shape[0]
    n_feature = LR_X.shape[1]
    print(n_sample, n_feature)
    # create train and validation data
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(LR_X, LR_y, test_size=0.1,
                                                                      random_state=RandomSeed)


    # initial model
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
    model_reg = AdaBoostRegressor(n_estimators=50)

    Wilder_list = ['Wilderness_Area' + str(i) for i in range(1, 5)]
    soil_list = ['Soil_Type' + str(i) for i in range(1, 41)]
    binary_list = Wilder_list + soil_list
    # a = dataset.iloc[:,0:30]
    # a_binary = a.loc[:,[i for i in a.columns if i in binary_list]]
    # a_conti = a.loc[:,[i for i in a.columns if i not in binary_list]]

    N_feature = X_train.shape[1]  # feature number
    N_sample = X_train.shape[0]  # feature length,i.e., sample number

    # ======================================================================================================================

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # DQN

    BATCH_SIZE = 32
    LR = 0.01
    EPSILON = 0.9
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 100  # After how much time you refresh target network
    MEMORY_CAPACITY = 400  # The size of experience replay buffer
    EXPLORE_STEPS = max(N_feature * 10,
                        1000)  # How many exploration steps you'd like, should be larger than MEMORY_CAPACITY
    N_ACTIONS = 2
    # N_STATES = env.observation_space.shape[0]
    N_STATES = len(X_train)

    np.random.seed(Random_SEED)
    torch.manual_seed(Random_SEED)  # reproducible

    action_list = np.random.randint(2, size=N_feature)

    i = 0
    while sum(action_list) < 2:
        np.random.seed(i)
        action_list = np.random.randint(2, size=N_feature)
        i += 1

    X_selected = X_train[:, action_list == 1]
    s = Feature_GCN(X_selected)

    if dataset_name in reg_name:
        model_reg.fit(X_train[:, action_list == 1], Y_train)
        pred_y = model_reg.predict(X_val[:, action_list == 1])
        mse = mean_squared_error(Y_val, pred_y)
        ave_corr = pd.DataFrame(X_val).corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
        r_list = (mse - 10 * ave_corr) / sum(action_list) * action_list

    elif dataset_name in cls_name:
        model.fit(X_train.iloc[:, action_list == 1], Y_train)
        accuracy = model.score(X_val.iloc[:, action_list == 1], Y_val)
        ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
        r_list = (accuracy - 10 * ave_corr) / sum(action_list) * action_list
    action_list_p = action_list
    dqn_list = []
    for agent in range(N_feature):
        dqn_list.append(DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS))
    # The element in the result list consists two parts,
    # i.e., accuracy and the action list (action 1 means selecting corresponding feature, 0 means deselection).
    result = []


    for i in range(EXPLORE_STEPS):
        action_list = np.zeros(N_feature)
        for agent, dqn in enumerate(dqn_list):
            action_list[agent] = dqn.choose_action(s)

        while sum(action_list) < 2:
            np.random.seed(i)
            action_list = np.random.randint(2, size=N_feature)
            i += 1

        X_selected = X_train[:, action_list == 1]
        s_ = Feature_GCN(X_selected)

        if dataset_name in reg_name:
            model_reg.fit(X_train[:, action_list == 1], Y_train)
            pred_y = model_reg.predict(X_val[:, action_list == 1])
            mse = mean_squared_error(Y_val, pred_y)
            ave_corr = pd.DataFrame(X_val).corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
            action_list_change = np.array([x or y for (x, y) in zip(action_list_p, action_list)])
            r_list = (mse - 10 * ave_corr) / sum(action_list_change) * action_list_change

        elif dataset_name in cls_name:
            model.fit(X_train.iloc[:, action_list == 1], Y_train)
            accuracy = model.score(X_val.iloc[:, action_list == 1], Y_val)
            ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
            action_list_change = np.array([x or y for (x, y) in zip(action_list_p, action_list)])
            r_list = (accuracy - 10 * ave_corr) / sum(action_list_change) * action_list_change

        for agent, dqn in enumerate(dqn_list):
            dqn.store_transition(s, action_list[agent], r_list[agent], s_)

        if dqn_list[0].memory_counter > MEMORY_CAPACITY:
            for dqn in dqn_list:
                dqn.learn()
        s = s_
        action_list_p = action_list
        if dataset_name in reg_name:
            result.append([mse, action_list])
        elif dataset_name in cls_name:
            result.append([accuracy, action_list])

    output = []
    name = []

    name.append("result types")

    if dataset_name in reg_name:
        output.append('MA with mse')
    elif dataset_name in cls_name:
        output.append('MA with acc')

    max_accuracy = 0
    min_mse = 100000000
    optimal_set = []
    for i in range(len(result)):
        if dataset_name in reg_name:
            name.append("MSE of the {}-th explore step".format(i))
            output.append(result[i][0])
            if result[i][0] < min_mse:
                min_mse = result[i][0]
                optimal_set = result[i][1]
        elif dataset_name in cls_name:
            name.append("ACC of the {}-th explore step".format(i))
            output.append(result[i][0])
            if result[i][0] > max_accuracy:
                max_accuracy = result[i][0]
                optimal_set = result[i][1]

    if dataset_name in reg_name:
        print("The min_mse is: {}, the optimal selection for each feature is:{}".format(min_mse, optimal_set))
        name.append("feature subset")
        output.append(optimal_set)
        name.append("min_mse")
        output.append(min_mse)
    elif dataset_name in cls_name:
        print("The maximum accuracy is: {}, the optimal selection for each feature is:{}".format(max_accuracy,
                                                                                                 optimal_set))
        name.append("feature subset")
        output.append(optimal_set)
        name.append("max_accuracy")
        output.append(max_accuracy)


    out_1 = dict(zip(name, output))
    out_1 = pd.DataFrame([out_1])
    result_folder = './result/knockoff/baseline/'
    out_1.to_csv(result_folder + 'baseline_' + dataset_name + '_test.csv', mode='a')
