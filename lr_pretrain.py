import numpy as np
# import pandas as pd
# from sklearn import model_selection
# from sklearn.ensemble import RandomForestClassifier
# from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import Representation_learning
# import random
from sklearn.datasets import fetch_california_housing
from sklearn import ensemble
# from sklearn.metrics import mean_squared_error
# import knockpy.metro
# import warnings
# import statistics



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
# X_train, X_val, Y_train, Y_val = model_selection.train_test_split(LR_X, LR_y, test_size=0.1, random_state=RandomSeed)

# rhos = np.random.randn(n_feature)

# def log_likelihood(input_X):
#     return np.sum(input_X[:, 0:-1] * rhos[0:-1] * np.abs(input_X[:, 1:]))

# U = np.zeros((n_feature, n_feature))
# for xcoord in range(n_feature):
#     for offset in [-2, 1, 0, 1, 2]:
#         ycoord = min(max(0, xcoord + offset), n_feature - 1)
#         U[xcoord, ycoord] = 1

# warnings.filterwarnings("ignore")
#
# GaussianSampler = knockpy.knockoffs.GaussianSampler(LR_X)
# Xk = GaussianSampler.sample_knockoffs()

# metrosampler = knockpy.metro.MetropolizedKnockoffSampler(log_likelihood, X=LR_X, undir_graph=U)
# Xk = metrosampler.sample_knockoffs()

# X_knockoff = Xk
# ______________________________________________________________________________________________________________________
#
# feature_relationships = []
# feature_index = []
#
measure_type = 'Euclidean'
# # measure_type = 'Angle'
threshold_type = 'mean'
# # threshold_type = 'median'
# for i in range(n_feature):
#     current_feature = LR_X[:, i]
#     # print(current_feature)
#     # print(current_feature.shape)
#     # measure the relationship between current feature and knockoff matrix
#     # Euclidean Distance
#     if measure_type == 'Euclidean':
#         distances = np.linalg.norm(X_knockoff.T - current_feature, axis=1, ord=2)
#         print(np.shape(distances))
#         current_relationship = sum(distances)
#     # angle
#     # elif measure_type == 'Angle':
#     #     angles = [angle_between(X_knockoff[:, index], current_feature) for index in range(n_feature)]
#     #     current_relationship = sum(angles)
#     feature_relationships.append(current_relationship)
#     feature_index.append(i)
# print(feature_relationships)
#
# mean_value = np.mean(feature_relationships)
# median_value = statistics.median(feature_relationships)
#
# if threshold_type == 'mean':
#     threshold = mean_value
# elif threshold_type == 'median':
#     threshold = median_value
# knockoff_labels = []
# for i in range(n_feature):
#     if feature_relationships[i] > threshold:
#         knockoff_label = 1
#     else:
#         knockoff_label = 0
#     knockoff_labels.append(knockoff_label)
# print(knockoff_labels)

npy_save_path = './data/knockoff/knockoff_result/label/'
# np.save(npy_save_path + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy', knockoff_labels)

# ======================================================================================================================
knockoff_label = np.load(npy_save_path + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy')
print('knockoff label shape is: ', knockoff_label.shape)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# pre train the DQN decision making network
# create structure of decision making network
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

class newNet(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(newNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 1000)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same resultself.fc1 = nn.Linear(N_STATES, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.fc3 = nn.Linear(200, 50)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value

N_feature = n_feature
N_ACTIONS = 2
N_STATES = 480 + N_feature
train_epoch = 1000
y = torch.LongTensor(knockoff_label)
# print(y[53])

decision_network = Net(N_STATES, N_ACTIONS)
# decision_network = newNet(N_STATES, N_ACTIONS)


optimizer = torch.optim.SGD(decision_network.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

T = N_feature
correct_sum = 0
save_file = './data/knockoff/knockoff_result/model/train_result.txt'
file = open(save_file, mode='a')
print('mtro_' + dataset_name + '_' + measure_type + '_' + threshold_type, file=file)
for i in range(N_feature * train_epoch):
    t = i % T
    # X_selected = LR_X
    # X_array = np.array(X_selected)
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # X_array = min_max_scaler.fit_transform(X_array.reshape(-1, 1))
    # X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
    # s = Representation_learning.representation_training(X_tensor)
    # s = s.detach().numpy().reshape(-1)

    X_current = LR_X[:, t]
    X_current_array = np.array(X_current)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_current_array = min_max_scaler.fit_transform(X_current_array.reshape(-1, 1))
    X_current_tensor = torch.FloatTensor(X_current_array).unsqueeze(0).unsqueeze(0)
    s = Representation_learning.representation_training(X_current_tensor)
    s = s.detach().numpy().reshape(-1)
    print(s.shape)

    position = np.arange(1, N_feature + 1)
    onehot_encoded = OneHotEncoder(sparse=False).fit_transform(position.reshape(-1, 1))
    s = np.append(s, onehot_encoded[0])
    x = torch.unsqueeze(torch.FloatTensor(s), 0)

    action_value = decision_network.forward(x)
    # print(action_value)
    action = torch.max(action_value, 1)[1].data.numpy()
    action_choose = action[0]
    # print(action_value.type, y[t].type)
    loss = loss_func(action_value, y[t].view(-1))
    # print(loss)
    # print(t)
    # print(i)
    if t == 0 and i != 0:
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        acc = correct_sum / N_feature
        epoch_time = int(i / T)
        print('the accuracy of No.{} round is {}'.format(epoch_time, acc), file=open(save_file, 'a'))
        # print('the accuracy of No.{} round is {}'.format(epoch_time, acc))

        # print(epoch_time)
        # print(acc)
        correct_sum = 0
    if action_choose == y[t]:
        correct_sum = correct_sum + 1
        # print(correct_sum)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# ======================================================================================================================
model_save_folder = './data/knockoff/knockoff_result/model/'
model_save_path = model_save_folder + 'mtro_' + dataset_name + '_' + measure_type + '_' + threshold_type + '.pth'
torch.save(decision_network.state_dict(), model_save_path)

# path = "myFirstModel.pth"
# model.load_state_dict(torch.load(path))
