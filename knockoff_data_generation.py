import knockpy.metro
import numpy as np
import warnings
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # 根据原始数据生成Knockoff矩阵
#
# # input data
# # data_folder = './data/original/new_dataset_1/classification/'
# # cls_name = ['AP_Omentum_Ovary', 'german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
# #             'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
# #             'wine_red', 'wine_white', 'yeast', 'HumanActivity']
# data_with_ID = ['AP_Omentum_Ovary', 'uci_credit_card']
# data_folder = './data/original/new_dataset_1/regression/'
# reg_name = ['airfoil', 'bike_share', 'blogData', 'housing_boston', 'openml_586', 'openml_589', 'openml_607',
#             'openml_616', 'openml_618', 'openml_620', 'openml_637']
#
#
# # data_folder = './data/'
# # dataset = pd.read_csv(data_folder + 'short_train_Carto.csv')
# # dataset = pd.read_csv(data_folder + 'train_Carto.csv') # 第一列序号，54个特征，最后一列为标签，7分类
# # dataset = pd.read_csv(data_folder + 'phpDYCOet.csv') # 无序号，118个特征，最后一列为标签，2分类
# # dataset = pd.read_csv(data_folder + 'train_Amazon.csv') # 无序号，第一列为标签，9个特征，2分类
# # dataset = pd.read_csv(data_folder + 'train_cs.csv') # 第一列序号，第二列为标签，10个特征，2分类
# # dataset = pd.read_csv(data_folder + 'Glycation.csv') # 第一列序号，402个特征，最后一列为标签，2分类
# # dataset = pd.read_csv(data_folder + 'blogData_train.csv')
#
# def log_likelihood(input_X):
#     return np.sum(input_X[:, 0:-1] * rhos[0:-1] * np.abs(input_X[:, 1:]))
#
#
# for dataset_name in reg_name:
#
#     dataset_path = data_folder + dataset_name + '.csv'
#     dataset = pd.read_csv(dataset_path)
#     # print(dataset)
#
#     if dataset_name in data_with_ID:
#         dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
#     # rem = ['Id']
#     # dataset.drop(rem, axis=1, inplace=True)
#     r, c = dataset.shape
#     print(r, c)
#     # array = dataset.values
#     X = np.array(dataset.iloc[:, 0:(c - 1)])
#     print(X)
#     print(X.shape)
#     Y = np.array(dataset.iloc[:, (c - 1)])
#     print(Y)
#     p = c - 1
#     n = r
#
#     # An arbitrary (unnormalized) log-likelihood function
#     rhos = np.random.randn(p)
#
#     # Undirected graph
#
#     U = np.zeros((p, p))
#     for xcoord in range(p):
#         for offset in [-2, 1, 0, 1, 2]:
#             ycoord = min(max(0, xcoord + offset), p - 1)
#             U[xcoord, ycoord] = 1
#
#     warnings.filterwarnings("ignore")
#     # metrosampler = knockpy.metro.MetropolizedKnockoffSampler(log_likelihood, X=X, undir_graph=U)
#     # Xk = metrosampler.sample_knockoffs()
#     # FXSampler = knockpy.knockoffs.FXSampler(X)
#     # Xk = FXSampler.sample_knockoffs()
#     GaussianSampler = knockpy.knockoffs.GaussianSampler(X)
#     Xk = GaussianSampler.sample_knockoffs()
#     # BaseSampler = knockpy.knockoffs.KnockoffSampler()
#     # Xk = BaseSampler.sample_knockoffs()
#     # Xk = knockpy.knockoffs.produce_FX_knockoffs(X,)
#     # kfilter = knockpy.knockoff_filter.KnockoffFilter(
#     #     fstat='lcd',
#     #     ksampler='gaussian',
#     #     knockoff_kwargs={"method":"mvr"}
#     # )
#     # Xk = kfilter.sample_knockoffs()
#
#     # print(Xk)
#     print(Xk.shape)
#     # print(type(Xk))
#     df_Xk = pd.DataFrame(Xk)
#     dataset_new = pd.concat([dataset, df_Xk], axis=1)
#     print(dataset_new.shape)
#     data_save_name = './data/knockoff/new/Gaussian_' + dataset_name + '.csv'
#     # dataset_new.to_csv(path_or_buf='./data/knockoff/Gaussian_blogData.csv', index=False)
#     dataset_new.to_csv(path_or_buf=data_save_name, index=False)
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 根据Knockoff矩阵和原始特征之间的相关性选择‘bad’特征
# input data
# data_folder = './data/knockoff/'
# dataset_name = 'Gaussian_shortCarto'
# dataset_name = 'metro_shortCarto'
# dataset_name = 'Gaussian_Carto'  # 第一列序号，54个特征，7分类标签，54个knockoff伪特征
# dataset_name = 'metro_Carto'  # 第一列序号，54个特征，7分类标签，54个knockoff伪特征
# dataset_name = 'Gaussian_phpDYCOet'  # 无序号，118个特征，2分类标签，118个knockoff伪特征
# dataset_name = 'Gaussian_Amazon'  # 无序号，第一列为2分类标签，9个特征，9个knockoff伪特征
# dataset_name = 'metro_Amazon'  # 无序号，第一列为2分类标签，9个特征，9个knockoff伪特征
# dataset_name = 'Gaussian_cs'  # 无序号，第一列为2分类标签，10个特征，10个knockoff伪特征
# dataset_name = 'metro_cs'  # 无序号，第一列为2分类标签，10个特征，10个knockoff伪特征
# dataset_name = 'Gaussian_Glycation'  # 无序号，402个特征，2分类标签，402个knockoff伪特征
# dataset_name = 'metro_Glycation'  # 无序号，402个特征，2分类标签，402个knockoff伪特征
# dataset_name = 'Gaussian_blogData'  # 无序号，402个特征，2分类标签，402个knockoff伪特征

data_folder = './data/knockoff/new/Gaussian_'
data_with_ID = ['AP_Omentum_Ovary','uci_credit_card']
# cls_name = ['AP_Omentum_Ovary', 'german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
#             'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
#             'wine_red', 'wine_white', 'yeast', 'HumanActivity']
cls_name = ['german_credit']
# reg_name = ['airfoil', 'bike_share', 'blogData', 'housing_boston', 'openml_586', 'openml_589', 'openml_607',
#             'openml_616', 'openml_618', 'openml_620', 'openml_637']
reg_name = ['housing_boston']
dataset_names = cls_name + reg_name

for dataset_name in dataset_names:

    dataset = pd.read_csv(data_folder + dataset_name + '.csv')
    r, c = dataset.shape
    print(r, c)
    n_sample = r
    n_feature = int((c - 1) / 2)
    print('n_feature:', n_feature)
    print('n_sample:', n_sample)
    array = dataset.values
    X = np.array(dataset.iloc[:, 0:n_feature])
    # print(X)
    print('X shape:', X.shape)
    Y = np.array(dataset.iloc[:, n_feature])
    # print(Y)
    print('Y shape:', Y.shape)
    X_knockoff = np.array(dataset.iloc[:, n_feature + 1:c])
    # print(X_knockoff)
    print('X knockoff shape:', X_knockoff.shape)

    feature_relationships = []
    feature_index = []

    measure_type = 'Euclidean'
    # threshold_type = 'mean'
    threshold_type = 'median'
    for i in range(n_feature):
        current_feature = X[:, i]
        # measure the relationship between current feature and knockoff matrix
        # Euclidean Distance
        distances = np.linalg.norm(X_knockoff.T - current_feature, axis=1, ord=2)
        print(np.shape(distances))
        current_relationship = sum(distances)
        feature_relationships.append(current_relationship)
        feature_index.append(i)
    print(feature_relationships)

    mean_value = np.mean(feature_relationships)
    median_value = statistics.median(feature_relationships)

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
    #
    # npy_save_path = './data/knockoff/knockoff_result/label/'
    # np.save(npy_save_path + dataset_name + '_' + measure_type + '_' + threshold_type + '.npy', knockoff_labels)

    # visualization

    # # plt.rcParams["font.sans-serif"] = ['SimHei']
    # plt.rcParams["axes.unicode_minus"] = False
    # for i in range(len(feature_relationships)):
    #     plt.bar(feature_index[i], feature_relationships[i])
    # plt.plot([-2, n_feature + 2], [mean_value, mean_value], c='b', linestyle='--', label='Mean')
    # plt.plot([-2, n_feature + 2], [median_value, median_value], c='r', linestyle='--', label='Median')
    # plt.yticks([])
    # plt.title('Relationship between Features and Knocoffs Matrix')
    # plt.xlabel("Feature Index")
    # plt.ylabel("Relationship")
    # plt.legend()
    # plt_save_path = './data/knockoff/knockoff_result/figure/'
    # plt.savefig(plt_save_path + dataset_name + '_' + measure_type + '.png')
    # plt.show()


    plt.style.use('seaborn-darkgrid')  # Enhanced plot style
    plt.rcParams["axes.unicode_minus"] = False

    # cmap = plt.get_cmap('copper')  # Warm color map for bars
    # colors = cmap(np.linspace(0, 1, len(feature_relationships)))
    bar_colors = ['#A52A2A', '#808000', '#B8860B', '#FF8C00', '#BC8F8F','#CD5C5C', '#DEB887', '#F4A460', '#DAA520', '#8B4513']  # 暖色调低饱和度颜色

    for i in range(len(feature_relationships)):
        plt.bar(feature_index[i], feature_relationships[i], color=bar_colors[i % len(bar_colors)])

    # Use desaturated colors for mean and median lines
    median_color = '#008B8B'  # 暗青色
    mean_color = '#CD5C5C'  # 柔和的赤褐色

    plt.plot([-2, n_feature + 2], [mean_value, mean_value], c=mean_color, linestyle='--', linewidth=2, label='Mean')
    plt.plot([-2, n_feature + 2], [median_value, median_value], c=median_color, linestyle='--', linewidth=2,
             label='Median')

    plt.yticks([])
    plt.title('Relationship between Features and Knockoffs Matrix', fontsize=14)
    plt.xlabel("Feature Index", fontsize=12)
    plt.ylabel("Relationship", fontsize=12)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt_save_path = './data/knockoff/knockoff_result/figure/'
    plt.savefig(plt_save_path + 'new_' + dataset_name + '_' + measure_type + '.png')
    plt.show()
