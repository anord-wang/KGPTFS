# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
import Representation_learning
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# input data and useful files
Random_SEED = 1998
data_folder = './data/original/'
knockoff_folder = './data/knockoff/'
knockoff_label_folder = './data/knockoff/knockoff_result/label/'
data_label_first = ['Amazon', 'cs', ]
mat_file_folder = './data/search_order/'

# choose dataset
# dataset_name = 'shortCarto'
dataset_name = 'Carto'
# dataset_name = 'phpDYCOet'
# dataset_name = 'Amazon'
# dataset_name = 'cs'
# dataset_name = 'Glycation'

# choose knockoff type
knockoff_type = 'metro'
# knockoff_type = 'Gaussian'

# choose measurement method
measure_type = 'Euclidean'

# choose threshold type
threshold_type = 'mean'
# threshold_type = 'median'

# input data
dataset_all = pd.read_csv(knockoff_folder + knockoff_type + '_' + dataset_name + '.csv')

# input knockoff label
knockoff_label = np.load(
    knockoff_label_folder + knockoff_type + '_' + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy')

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
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_original, Y, test_size=0.1,
                                                                  random_state=Random_SEED)

# initial model
model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=0)
# model =DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)

# load feature order
mat_file_name = mat_file_folder + 'IGorder' + dataset_name + '.mat'
Forder = loadmat(mat_file_name)
order = Forder["Forder"].squeeze(0) - 1

# put features in order
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
EXPLORE_STEPS = 10 * N_sample  # How many exploration steps you'd like, should be larger than MEMORY_CAPACITY20
VAL_STEPS = N_sample
N_ACTIONS = 2
N_STATES = 960 + N_feature
knockoff_p = 0.7


#

class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(N_STATES, 100)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        # self.out = nn.Linear(100, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)  # initialization
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


class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS, knockoff_label=None, model_path=None):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        # state_dict = torch.load(model_path)
        # self.eval_net.load_state_dict(state_dict)
        # self.target_net.load_state_dict(state_dict)

        self.learn_step_counter = 0
        self.EPSILON = EPSILON
        self.knockoff_EPSILON = knockoff_EPSILON
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON_STEP_SIZE = EPSILON_STEP_SIZE
        self.knockoff_label = knockoff_label[order]

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


np.random.seed(Random_SEED)
torch.manual_seed(Random_SEED)  # reproducible

model_save_folder = './data/knockoff/knockoff_result/model/'
current_path = model_save_folder + knockoff_type + '_' + dataset_name + '_' + measure_type + '_' + threshold_type + '_newnet.pth'
dqn = DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS, knockoff_label=knockoff_label, model_path=current_path)
# dqn = DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS)
# # The element in the result list consists two parts,
# # i.e., accuracy and the action list (action 1 means selecting corresponding feature, 0 means deselection).
#

L1 = random.sample(range(0, N_feature), 2)
Fstate = np.zeros(N_feature)
Fstate[[index for index in L1]] = 1

X_selected = X_train.iloc[:, Fstate == 1]
X_array = np.array(X_selected)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X_array = min_max_scaler.fit_transform(X_array)
X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
s = Representation_learning.representation_training(X_tensor)
s = s.detach().numpy().reshape(-1)
print(s.shape)

X_current = X_train.iloc[:, 0]
X_current_array = np.array(X_current)
X_current_array = min_max_scaler.fit_transform(X_current_array.reshape(-1, 1))
X_current_tensor = torch.FloatTensor(X_current_array).unsqueeze(0).unsqueeze(0)
s_current = Representation_learning.representation_training(X_current_tensor)
s_current = s_current.detach().numpy().reshape(-1)
s = np.append(s, s_current)
print(s.shape)

position = np.arange(1, N_feature + 1)
onehot_encoded = OneHotEncoder(sparse=False).fit_transform(position.reshape(-1, 1))
s = np.append(s, onehot_encoded[0])
print(s.shape)
knockoff_select_list = np.zeros(N_feature)

result = []
T = N_feature
# dqn.EPSILON_STEP_SIZE = (dqn.EPSILON_MAX - dqn.EPSILON)/((EXPLORE_STEPS-1)*T)
for i in range(EXPLORE_STEPS):
    t = i % T
    # Faction = dqn.choose_action(s)
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

    X_current = X_train.iloc[:, t]
    X_current_array = np.array(X_current)
    X_current_array = min_max_scaler.fit_transform(X_current_array.reshape(-1, 1))
    X_current_tensor = torch.FloatTensor(X_current_array).unsqueeze(0).unsqueeze(0)
    s_current_ = Representation_learning.representation_training(X_current_tensor)
    s_current_ = s_current_.detach().numpy().reshape(-1)
    s_ = np.append(s_, s_current_)

    if t == T - 1:
        s_ = np.append(s_, onehot_encoded[0])
    else:
        s_ = np.append(s_, onehot_encoded[t + 1])

    X_selected = X_train.iloc[:, Fstate == 1]

    X_train_array = np.array(X_train)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_train_array = min_max_scaler.fit_transform(X_train_array)
    X_train_tensor = torch.FloatTensor(X_train_array).unsqueeze(0).unsqueeze(0)

    X_selected_array = np.array(X_selected)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_selected_array = min_max_scaler.fit_transform(X_selected_array)
    X_selected_array = np.pad(X_selected_array, ((0, 0), (0, X_train_array.shape[1] - X_selected_array.shape[1])),
                              'constant', constant_values=(0, 0))
    X_selected_tensor = torch.FloatTensor(X_selected_array).unsqueeze(0).unsqueeze(0)

    reconstruction_loss = Representation_learning.reconstruction_training(X_selected_tensor, X_train_tensor)

    corr = X_val.corr().abs()
    ave_corr = (corr.iloc[:, t].sum()) / (X_val.shape[1])

    feature_list = s[N_STATES - N_feature:N_STATES]
    feature_index = [index for index, e in enumerate(feature_list) if e == 1]
    knockoff_para = knockoff_label[order][feature_index]
    knockoff_likehood = abs(np.float64(Factionvalue[1]))
    knockoff_reward = (knockoff_p ** int(knockoff_select_list[feature_index])) * int(
        1 - knockoff_para) * Faction * knockoff_likehood
    knockoff_select_list[feature_index] = knockoff_select_list[feature_index] + Faction

    # ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
    # r = (accuracy - knockoff_reward - ave_corr)
    r = (1 - reconstruction_loss.detach().numpy() + knockoff_reward - ave_corr)
    # r = accuracy

    dqn.store_transition(s, Faction, r, s_)

    # 每100步计算一次acc进行保存
    if (i != 0) and (i % VAL_STEPS == 0):
        model.fit(X_train.iloc[:, Fstate == 1], Y_train)
        accuracy = model.score(X_val.iloc[:, Fstate == 1], Y_val)
        Y_pred = model.predict(X_val.iloc[:, Fstate == 1])
        macroF1 = f1_score(Y_val, Y_pred, average='macro')
        precision = precision_score(Y_val, Y_pred, average='macro')
        recall = recall_score(Y_val, Y_pred, average='macro')
        print(r, accuracy)
        result.append([accuracy, precision, recall, macroF1, Fstate, r])

    dqn.store_transition(s, Faction, r, s_)

    if dqn.memory_counter > MEMORY_CAPACITY:
        dqn.learn()
    s = s_

output = []
reward_output = []
name = []
name.append("result types")
output.append('new network')
reward_output.append('new network')

max_accuracy = 0
optimal_set = []
for i in range(len(result)):
    name.append("Accuracy of the {}-th explore step".format(i*VAL_STEPS))
    output.append(result[i][0])
    reward_output.append(result[i][5])

    if result[i][0] > max_accuracy:
        max_accuracy = result[i][0]
        optimal_set = result[i][4]
        Mmacro_f1 = result[i][3]
        Mprecision = result[i][1]
        Mrecall = result[i][2]

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
out_1.to_csv(result_folder + dataset_name + '_test.csv', mode='a')
out_2.to_csv(result_folder + dataset_name + '_reward_test.csv', mode='a')