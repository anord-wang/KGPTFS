# Import SD and numpy.
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.io import arff
import numpy as np

# fileName_1 = './data/GRFG_data/train.csv'
# fileName_2 = './data/GRFG_data/test.csv'
# out_dir = './data/GRFG_data/HumanActivity.csv'
# #
# # df = pd.read_hdf(fileName)
# df_1 = pd.read_csv(fileName_1)
# df_2 = pd.read_csv(fileName_2)
# df_1 = pd.DataFrame(df_1)
# df_2 = pd.DataFrame(df_2)
# print(df_1.shape)
# print(df_2.shape)
# df_all = pd.concat([df_1, df_2], axis=0)
# print(df_all.shape)
# le = LabelEncoder()
# le.fit(df_all['Activity'])
#
# df_all['Activity'] = le.transform(df_all['Activity'])
# # df_all['Activity'] = pd.get_dummies(df_all['Activity'])
# df_all.to_csv(out_dir, index=False)

# file_name = './data/GRFG_data/Mice-Protein.xls'

# data, meta = arff.loadarff(file_name)
# print(data)
# print(meta)

# df = pd.DataFrame(data)
# print(df.head())
# print(df.shape)
# df.to_csv(out_dir, index=False)


data_folder = './data/knockoff/new/Gaussian_'
data_with_ID = ['AP_Omentum_Ovary', 'uci_credit_card']
cls_name = ['AP_Omentum_Ovary', 'german_credit', 'higgs', 'ionosphere', 'lymphography', 'mammography',
            'messidor_features', 'pima_indian', 'spam_base', 'spectf', 'svmguide3', 'uci_credit_card', 'wbc',
            'wine_red', 'wine_white', 'yeast', 'HumanActivity','cs','phpDYCOet']
reg_name = ['airfoil', 'bike_share', 'blogData', 'housing_boston', 'openml_586', 'openml_589', 'openml_607',
            'openml_616', 'openml_618', 'openml_620', 'openml_637']

for dataset_name in cls_name + reg_name:
    dataset = pd.read_csv(data_folder + dataset_name + '.csv')
    r, c = dataset.shape
    n_sample = r
    n_feature = int((c - 1) / 2)
    print('dataset name:', dataset_name)
    print('n_feature:', n_feature)
    print('n_sample:', n_sample)
