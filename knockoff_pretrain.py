# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import Representation_learning

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# input data and useful files
Random_SEED = 2023
data_folder = './data/original/'
knockoff_folder = './data/knockoff/new/'
knockoff_label_folder = './data/knockoff/knockoff_result/label/'
data_label_first = ['Amazon', 'cs', ]
mat_file_folder = './data/search_order/'

# choose dataset
# dataset_name = 'shortCarto'
# dataset_name = 'Carto'
# dataset_name = 'phpDYCOet'
# dataset_name = 'Amazon'
# dataset_name = 'cs'
# dataset_name = 'Glycation'

# data_folder = './data/original/new_dataset_1/classification/'
cls_name = ['AP_Omentum_Ovary', 'german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
            'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
            'wine_red', 'wine_white', 'yeast', 'HumanActivity', 'cs', 'phpDYCOet']
data_with_ID = ['AP_Omentum_Ovary', 'uci_credit_card']
# data_folder = './data/original/new_dataset_1/regression/'
reg_name = ['airfoil', 'bike_share', 'blogData', 'housing_boston', 'openml_586', 'openml_589', 'openml_607',
            'openml_616', 'openml_618', 'openml_620', 'openml_637']
name_list = cls_name + reg_name
order_list = ['cs', 'phpDYCOet']


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
        self.fc1.weight.data.normal_(0,
                                     0.1)  # initialization, set seed to ensure the same resultself.fc1 = nn.Linear(N_STATES, 1000)
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
    # knockoff_label = np.load(
    #     knockoff_label_folder + knockoff_type + '_' + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy')
    knockoff_label = np.load(knockoff_label_folder + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy')
    print('knockoff label shape is: ', knockoff_label.shape)

    # get information
    r, c = dataset_all.shape
    n_sample = r
    n_feature = int((c - 1) / 2)

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
    X_original_train, X_original_val, X_knockoff_train, X_knockoff_val, Y_train, Y_val = model_selection.train_test_split(
        X_original, X_knockoff, Y, test_size=0.1, random_state=Random_SEED)

    # initial model
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
    # model =DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)

    if dataset_name in order_list:
        mat_file_name = mat_file_folder + 'IGorder' + dataset_name + '.mat'
        Forder = loadmat(mat_file_name)
        order = Forder["Forder"].squeeze(0) - 1

        X_original_train = X_original_train.iloc[:, order]
        X_original_val = X_original_val.iloc[:, order]
        X_knockoff_train = X_knockoff_train.iloc[:, order]
        X_knockoff_val = X_knockoff_val.iloc[:, order]

    N_feature = X_original_train.shape[1]  # feature number
    N_sample = X_original_val.shape[0]  # feature length,i.e., sample number

    # ======================================================================================================================

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    print(knockoff_type + '_' + dataset_name + '_' + measure_type + '_' + threshold_type, file=file)
    for i in range(N_feature * train_epoch):
        t = i % T
        # X_selected = X_original_train
        # X_array = np.array(X_selected)
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # X_array = min_max_scaler.fit_transform(X_array.reshape(-1, 1))
        # X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
        # s = Representation_learning.representation_training(X_tensor)
        # s = s.detach().numpy().reshape(-1)

        X_current = X_original_train.iloc[:, t]
        X_current_array = np.array(X_current)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        X_current_array = min_max_scaler.fit_transform(X_current_array.reshape(-1, 1))
        X_current_tensor = torch.FloatTensor(X_current_array).unsqueeze(0).unsqueeze(0)
        s_current = Representation_learning.representation_training(X_current_tensor)
        s_current = s_current.detach().numpy().reshape(-1)
        # s = np.append(s, s_current)
        # print(s.shape)

        position = np.arange(1, N_feature + 1)
        onehot_encoded = OneHotEncoder(sparse=False).fit_transform(position.reshape(-1, 1))
        s = np.append(s_current, onehot_encoded[0])
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
    model_save_path = model_save_folder + knockoff_type + '_' + dataset_name + '_' + measure_type + '_' + threshold_type + '_newnet.pth'
    torch.save(decision_network.state_dict(), model_save_path)
