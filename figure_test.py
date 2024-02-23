import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#
# # table 8
# # Given data
# metrics = ['Glycation', 'Housing\nCalifornia']
# kgptfs = [4349.60, 947.23]  # Running time for KGPTFS
# sadrlfs = [8848.02, 1529.63]  # Running time for SADRLFS
#
# x = np.arange(len(metrics))
#
# # Colors
# colors = ['#e76f51', '#f4a261']  # Brick red and sand color
#
# # Create horizontal bar chart
# plt.figure(figsize=(14, 7))
# bar_width = 0.35
# bars1 = plt.barh(x + bar_width, kgptfs, height=bar_width, color=colors[0], label='KGPTFS')
# bars2 = plt.barh(x, sadrlfs, height=bar_width, color=colors[1], label='SADRLFS')
#
# # Label and title settings
# plt.xlabel('Running Time (s)', fontsize=18)
# plt.ylabel('Dataset', fontsize=18)
# plt.title('Comparison of Running Time Between KGPTFS and SADRLFS', fontsize=18)
# plt.yticks(x + bar_width / 2, metrics, fontsize=16)
# plt.xticks(fontsize=16)
# plt.legend(fontsize=16)
#
# # Axis style settings
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_color('grey')
# plt.gca().spines['bottom'].set_linewidth(0.5)
# plt.gca().spines['left'].set_color('grey')
# plt.gca().spines['left'].set_linewidth(0.5)
#
# # Adding value labels on top of the bars
# def add_labels(bars):
#     for bar in bars:
#         width = bar.get_width()
#         plt.text(bar.get_x() + width + 50, bar.get_y() + bar.get_height()/2,
#                  f'{width}', ha='left', va='center', fontsize=16)
#
# add_labels(bars1)
# add_labels(bars2)
# # 保存图片
# plt.savefig('./runningtime.png', bbox_inches='tight')
# # Display the plot
# plt.show()


# # table 7
# # 数据
# models_hc = ['Linear Regression', 'AdaBoost', 'K-Neighbors', 'Random Forest', 'Gradient Boosting', 'Extra Tree']
# l2_norm_hc = [0.6027, 0.6904, 0.7399, 0.6904, 0.6904, 0.8525]
#
# models_glc = ['Random Forest', 'Decision Tree', 'Gaussian Naive Bayes', 'MLP', 'AdaBoost', 'Gradient Boosting']
# acc_glc = [85.71, 85.71, 84.13, 87.30, 79.37, 74.60]
#
# # 将ACC数据转换为[0,1]区间内的值
# acc_glc = np.array(acc_glc) / 100.0
#
# # 计算雷达图的角度
# angles_hc = np.linspace(0, 2 * np.pi, len(models_hc), endpoint=False).tolist()
# angles_glc = np.linspace(0, 2 * np.pi, len(models_glc), endpoint=False).tolist()
#
# # 使图形闭合
# l2_norm_hc = np.concatenate((l2_norm_hc, [l2_norm_hc[0]]))
# acc_glc = np.concatenate((acc_glc, [acc_glc[0]]))
# angles_hc += angles_hc[:1]
# angles_glc += angles_glc[:1]
#
# # 创建两个子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))
#
# # 字体大小
# font_size = 16
#
# # 第一个子图 - Housing California 数据集
# ax1.fill(angles_hc, l2_norm_hc, alpha=0.25)
# ax1.set_xticks(angles_hc[:-1])
# ax1.set_xticklabels(models_hc, size=font_size)
# ax1.set_title('Housing California (l2-norm)', size=font_size + 2)
#
# # 第二个子图 - Glc 数据集
# ax2.fill(angles_glc, acc_glc, alpha=0.25)
# ax2.set_xticks(angles_glc[:-1])
# ax2.set_xticklabels(models_glc, size=font_size)
# ax2.set_title('Glycation (ACC)', size=font_size + 2)
#
# # 设置图例位置和大小
# ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=font_size)
# ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=font_size)
#
# # 调整子图之间的距离，确保字体不重叠
# plt.subplots_adjust(wspace=0.4)
#
# # 保存图片
# plt.savefig('./generalization.png', bbox_inches='tight')
#
# # 显示图形
# plt.show()

# # table 6
# 重新生成条形图，增加字体大小并在每个柱状图上标注对应的值

# 更新代码，确保两个子图的纵坐标尺度正确，并去掉网格线

# 数据准备
methods = ['Reconstruction\nComparison', 'Reconstruction\nLoss', 'KGPTFS w/o\nReconstruction']
acc_scores = [85.71, 84.13, 82.54]  # ACC scores for Glc
l2_norm_scores = [0.6027, 0.6904, 0.6168]  # l2-norm scores for Housing California

x = np.arange(len(methods))  # 方法标签的位置
width = 0.25  # 柱状图的宽度
font_size = 14  # 字体大小

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 第一个子图 - Glc 数据集的ACC
ax1.bar(x, acc_scores, width, color=['#6b705c', '#a5a58d', '#cb997e'])
ax1.set_title('ACC on Glycation Dataset', fontsize=font_size)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=font_size)
ax1.set_ylabel('Accuracy (%)', fontsize=font_size)
ax1.set_ylim(70, 90)  # 设置y轴范围为0到100
ax1.grid(False)  # 去掉网格线

# 在每个柱状图上方显示数值
for i in range(len(acc_scores)):
    ax1.text(i, acc_scores[i] + 1, f'{acc_scores[i]}%', ha='center', va='bottom', fontsize=font_size)

# 第二个子图 - Housing California 数据集的l2-norm
ax2.bar(x, l2_norm_scores, width, color=['#6b705c', '#a5a58d', '#cb997e'])
ax2.set_title('l2-norm on Housing California Dataset', fontsize=font_size)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=font_size)
ax2.set_ylabel('l2-norm', fontsize=font_size)
ax2.set_ylim(0.4, 0.8)  # 设置y轴范围为0到1
ax2.grid(False)  # 去掉网格线

# 在每个柱状图上方显示数值
for i in range(len(l2_norm_scores)):
    ax2.text(i, l2_norm_scores[i] + 0.02, f'{l2_norm_scores[i]:.4f}', ha='center', va='bottom', fontsize=font_size)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('./reconstruction.png', bbox_inches='tight')

# 显示图形
plt.show()

# 根据用户提供的表格样式，使用新的配色方案进行可视化

# # table 4
#
# # 数据
# methods = ['KGPTFS', 'KGPTFS \nw/o Pre-train', 'KGPTFS \nw/o Reward', 'KGPTFS \nw/o ε-greedy']
# acc_scores = [85.71, 82.54, 84.13, 82.54]  # ACC scores for Glc
# l2_norm_scores = [0.6027, 0.6904, 0.6509, 0.7399]  # l2-norm scores for Housing California
#
# x = np.arange(len(methods))  # 方法标签的位置
# width = 0.35  # 柱状图的宽度
# font_size = 14
# # 设置配色方案
# # 深蓝色、深红色、深绿色、深黄色
# colors = ['#6d6875', '#b5838d', '#e5989b', '#ffb4a2']
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
# # ACC 柱状图
# ax1.bar(x, acc_scores, width, color=colors)
# ax1.set_title('ACC on Glycation Dataset', fontsize=font_size)
# ax1.set_xticks(x)
# ax1.set_xticklabels(methods, fontsize=font_size)
# ax1.set_ylabel('Accuracy (%)', fontsize=font_size)
# ax1.set_ylim(70, 90)  # 设定y轴的范围
#
# # l2-norm 柱状图
# ax2.bar(x, l2_norm_scores, width, color=colors)
# ax2.set_title('l2-norm on Housing California Dataset', fontsize=font_size)
# ax2.set_xticks(x)
# # ax2.set_xticklabels(methods, ha='right')
# ax2.set_xticklabels(methods, fontsize=font_size)
#
# ax2.set_ylabel('l2-norm', fontsize=font_size)
# ax2.set_ylim(0.4, 0.8)  # 设定y轴的范围
#
# # 在每个柱状图上标注值
# for i in range(len(acc_scores)):
#     ax1.text(i, acc_scores[i] + 1, f"{acc_scores[i]}%", ha='center', va='bottom', fontsize=font_size)
#     ax2.text(i, l2_norm_scores[i] + 0.02, f"{l2_norm_scores[i]}", ha='center', va='bottom', fontsize=font_size)
#
# # 调整布局
# plt.tight_layout()
# # 保存图片
# plt.savefig('./knockoff.png', bbox_inches='tight')
# plt.show()


# # table 5
# # Replotting the line charts without rotated x-tick labels and with muted colors
#
# # Data from the table
# parameter_ratios = ['10:0:90', '0:10:90', '5:5:90', '10:10:80']
# l2_norm_scores = [0.7399, 0.6846, 0.6027, 0.7105]  # l2-norm for Housing California (reordered)
# acc_scores = [82.54, 79.37, 85.71, 84.13]  # ACC for Glc (reordered)
# font_size = 14
# # Normalize ACC scores to match the scale of l2-norm
# acc_scores_normalized = [score / 100 for score in acc_scores]
#
# # Colors
# colors = ['#7b6d8d', '#7d9a92']  # Muted blue and green
#
# # Plotting the line charts
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
# # Subplot 1: l2-norm for Housing California
#
# ax1.plot(parameter_ratios, l2_norm_scores, marker='o', color=colors[0], label='l2-norm (Housing California)')
# ax1.set_title('l2-norm on Housing California Dataset', fontsize=font_size)
# ax1.set_xlabel('Parameters Ratio', fontsize=font_size)
# ax1.set_ylabel('l2-norm', fontsize=font_size)
# ax1.set_xticks(parameter_ratios)
#
# # Subplot 2: ACC for Glc
# ax2.plot(parameter_ratios, acc_scores_normalized, marker='s', color=colors[1], label='ACC (Glycation)')
# ax2.set_title('ACC on Glycation Dataset', fontsize=font_size)
# ax2.set_xlabel('Parameters Ratio', fontsize=font_size)
# ax2.set_ylabel('ACC (Normalized)', fontsize=font_size)
# ax2.set_xticks(parameter_ratios)
#
# # Set y-axis limit to match the data range
# ax1.set_ylim(0.4, max(l2_norm_scores) + 0.1)
# ax2.set_ylim(0.6, max(acc_scores_normalized) + 0.1)
#
# # Show legends
# ax1.legend()
# ax2.legend()
#
# # Adjust the layout to prevent clipping of tick-labels
# plt.tight_layout()
# # 保存图片
# plt.savefig('./greedy.png', bbox_inches='tight')
# plt.show()


# # Data extracted from the uploaded table
# data = {
#     'Dataset': [
#         'Carto', 'Amazon Employee', 'Glycation', 'SpectF', 'German Credit', 'UCI Credit', 'SpamBase',
#         'Ionosphere', 'Activity', 'Higgs Boson', 'PimaIndian', 'Messidor Feature', 'Wine Quality Red',
#         'Wine Quality White', 'yeast', 'phpYCOet', 'Housing California', 'Housing Boston', 'Airfoil',
#         'Openml 618', 'Openml 589', 'Openml 616', 'Openml 607', 'Openml 620', 'Openml 637', 'Openml 586'
#     ],
#     'KGPTFS': [
#         88.10, 95.03, 85.71, 92.59, 82.00, 82.70, 97.61,
#         94.44, 98.44, 72.34, 77.92, 70.69, 83.00,
#         71.02, 91.28, 97.39, 0.6027, 9.5343, 4.3359,
#         0.1822, 0.1429, 0.1778, 0.1682, 0.1518, 0.2232, 0.2215
#     ],
#     'KGPTFS_with_Random_Reward': [
#         81.81, 94.66, 77.78, 85.19, 74.00, 79.17, 93.28,
#         88.89, 95.24, 63.56, 73.12, 66.38, 80.00,
#         71.43, 85.71, 97.09, 0.6985, 23.7225, 16.1104,
#         0.4998, 0.4799, 0.4650, 0.5837, 0.7684, 0.5222, 0.4899
#     ],
#     'All_Feature_No_Selection': [
#         86.84, 94.39, 76.19, 92.86, 78.00, 81.33, 96.75,
#         91.67, 97.67, 70.78, 81.82, 75.00, 71.00,
#         62.65, 89.26, 97.07, 0.7502, 8.3632, 4.4268,
#         0.2411, 0.1782, 0.2859, 0.3082, 0.1896, 0.2550, 0.2039
#     ]
# }
#
# # Convert the data into a Pandas DataFrame for easier plotting
# df = pd.DataFrame(data)
#
# # Filter the datasets into two groups: ACC and l2-norm
# acc_datasets = df['Dataset'].head(16)
# l2norm_datasets = df['Dataset'].tail(10)
#
# # Separate the data into ACC and l2-norm as well
# acc_data = df.iloc[:16, 1:].astype(float)
# l2norm_data = df.iloc[16:, 1:].astype(float)
#
# # Setting the colors for the bars
# colors = ['#e76f51', '#2a9d8f', '#264653']
# # colors = ['#6a4f4b', '#a39193', '#beb9b5']
#
# # Create subplots for ACC datasets
# fig_acc, axs_acc = plt.subplots(nrows=4, ncols=4, figsize=(20, 15), constrained_layout=True)
# axs_acc = axs_acc.flatten()
#
# aaaxxx = ['KGPTFS', 'Random Reward', 'All Feature']
# # Iterate over the ACC datasets to create bar charts
# for i, ax in enumerate(axs_acc.flatten()):
#     if i < len(acc_datasets):
#         # Data for the bars
#         acc_values = [acc_data['KGPTFS'].iloc[i] / 100, acc_data['KGPTFS_with_Random_Reward'].iloc[i] / 100,
#                       acc_data['All_Feature_No_Selection'].iloc[i] / 100]
#         half_max = max(acc_values) / 1.3
#         # Plotting the bars
#         bars = ax.bar(aaaxxx, acc_values, color=colors)
#         # Setting the title for each subplot
#         ax.set_title(acc_datasets.iloc[i], fontsize=14)
#         # Removing the x-axis labels for a cleaner look
#         ax.set_xticks(aaaxxx)
#         # Setting the y-axis limit based on the data
#         ax.set_ylim(half_max, 1.1 * max(acc_values))  # Assuming the ACC values are normalized
#         # Adding the data values on top of each bar
#         for bar in bars:
#             yval = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval*100:.2f}%', ha='center', va='bottom', fontsize=14)
#
#
# # Create subplots for l2-norm datasets
# fig_l2norm, axs_l2norm = plt.subplots(nrows=2, ncols=5, figsize=(20, 10), constrained_layout=True)
# axs_l2norm = axs_l2norm.flatten()
#
# # Continue creating l2-norm bar charts
# for i, ax in enumerate(axs_l2norm.flatten()):
#     if i < len(l2norm_datasets):
#         # Data for the bars
#         l2norm_values = [l2norm_data['KGPTFS'].iloc[i], l2norm_data['KGPTFS_with_Random_Reward'].iloc[i],
#                          l2norm_data['All_Feature_No_Selection'].iloc[i]]
#         # Plotting the bars
#         bars = ax.bar(aaaxxx, l2norm_values, color=colors)
#         # Setting the title for each subplot
#         ax.set_title(l2norm_datasets.iloc[i], fontsize=14)
#         # Removing the x-axis labels for a cleaner look
#         ax.set_xticks(aaaxxx)
#         # Setting the y-axis limit based on the data
#         ax.set_ylim(0, 1.1 * max(l2norm_values))
#         # Adding the data values on top of each bar
#         for bar in bars:
#             yval = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=14)
#     else:
#         # Hide the subplot if there is no data for it
#         ax.set_visible(False)
#
# # Adjust layout
# fig_acc.suptitle('ACC Scores for Different Datasets', fontsize=16)
# fig_l2norm.suptitle('l2-norm Scores for Different Datasets', fontsize=16)
#
# fig_acc.savefig('./ACCDifferent.png', bbox_inches='tight')
# fig_l2norm.savefig('./l2-normDifferent.png', bbox_inches='tight')
#
# plt.show()



# # Data extracted from the uploaded table
# data = {
#     'Dataset': [
#         'Carto', 'Amazon Employee', 'Glycation', 'SpectF', 'German Credit', 'UCI Credit', 'SpamBase',
#         'Ionosphere', 'Activity', 'Higgs Boson', 'PimaIndian', 'Messidor Feature', 'Wine Quality Red',
#         'Wine Quality White', 'yeast', 'phpYCOet', 'Housing California', 'Housing Boston', 'Airfoil',
#         'Openml 618', 'Openml 589', 'Openml 616', 'Openml 607', 'Openml 620', 'Openml 637', 'Openml 586'
#     ],
#     'KGPTFS': [
#         88.10, 95.03, 85.71, 92.59, 82.00, 82.70, 97.61,
#         94.44, 98.44, 72.34, 77.92, 70.69, 83.00,
#         71.02, 91.28, 97.39, 0.6027, 9.5343, 4.3359,
#         0.1822, 0.1429, 0.1778, 0.1682, 0.1518, 0.2232, 0.2215
#     ],
#     'KGPTFS_without_ReCONSTRACTION': [
#         87.83, 95.27, 82.54, 88.89, 78.00, 82.20, 97.18,
#         91.67, 98.05, 70.36, 81.82, 75.00, 80.00,
#         72.04, 90.60, 97.07, 0.6769, 9.7509, 3.5918,
#         0.2249, 0.1562, 0.2292, 0.2134, 0.1597, 0.2212, 0.1934
#     ],
#     'KGPTFS_without_Knockoff': [
#         87.89, 95.27, 84.13, 91.48, 72.00, 81.80, 96.31,
#         91.67, 98.35, 69.66, 67.53, 71.55, 75.00,
#         71.43, 86.58, 97.27, 0.6168, 9.3632, 5.5752,
#         0.2214, 0.1729, 0.2484, 0.2785, 0.1646, 0.1992, 0.1900
#     ]
# }
#
#
# # Convert the data into a Pandas DataFrame for easier plotting
# df = pd.DataFrame(data)
#
# # Filter the datasets into two groups: ACC and l2-norm
# acc_datasets = df['Dataset'].head(16)
# l2norm_datasets = df['Dataset'].tail(10)
#
# # Separate the data into ACC and l2-norm as well
# acc_data = df.iloc[:16, 1:].astype(float)
# l2norm_data = df.iloc[16:, 1:].astype(float)
#
# # Setting the colors for the bars
# # colors = ['#e76f51', '#2a9d8f', '#264653']
# colors = ['#a64b4f', '#cbb592', '#76637e']
#
# # Create subplots for ACC datasets
# fig_acc, axs_acc = plt.subplots(nrows=4, ncols=4, figsize=(20, 15), constrained_layout=True)
# axs_acc = axs_acc.flatten()
#
# aaaxxx = ['KGPTFS', 'KGPTFS w/o \n Reconstruction', 'KGPTFS w/o \n Knockoff']
# # Iterate over the ACC datasets to create bar charts
# for i, ax in enumerate(axs_acc.flatten()):
#     if i < len(acc_datasets):
#         # Data for the bars
#         acc_values = [acc_data['KGPTFS'].iloc[i] / 100, acc_data['KGPTFS_without_ReCONSTRACTION'].iloc[i] / 100,
#                       acc_data['KGPTFS_without_Knockoff'].iloc[i] / 100]
#         half_max = max(acc_values) / 1.3
#         # Plotting the bars
#         bars = ax.bar(aaaxxx, acc_values, color=colors)
#         # Setting the title for each subplot
#         ax.set_title(acc_datasets.iloc[i], fontsize=14)
#         # Removing the x-axis labels for a cleaner look
#         ax.set_xticks(aaaxxx)
#         # Setting the y-axis limit based on the data
#         ax.set_ylim(half_max, 1.1 * max(acc_values))  # Assuming the ACC values are normalized
#         # Adding the data values on top of each bar
#         for bar in bars:
#             yval = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval*100:.2f}%', ha='center', va='bottom', fontsize=14)
#
#
# # Create subplots for l2-norm datasets
# fig_l2norm, axs_l2norm = plt.subplots(nrows=2, ncols=5, figsize=(20, 10), constrained_layout=True)
# axs_l2norm = axs_l2norm.flatten()
#
# # Continue creating l2-norm bar charts
# for i, ax in enumerate(axs_l2norm.flatten()):
#     if i < len(l2norm_datasets):
#         # Data for the bars
#         l2norm_values = [l2norm_data['KGPTFS'].iloc[i], l2norm_data['KGPTFS_without_ReCONSTRACTION'].iloc[i],
#                          l2norm_data['KGPTFS_without_Knockoff'].iloc[i]]
#         # Plotting the bars
#         bars = ax.bar(aaaxxx, l2norm_values, color=colors)
#         # Setting the title for each subplot
#         ax.set_title(l2norm_datasets.iloc[i], fontsize=14)
#         # Removing the x-axis labels for a cleaner look
#         ax.set_xticks(aaaxxx)
#         # Setting the y-axis limit based on the data
#         ax.set_ylim(0, 1.1 * max(l2norm_values))
#         # Adding the data values on top of each bar
#         for bar in bars:
#             yval = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=14)
#     else:
#         # Hide the subplot if there is no data for it
#         ax.set_visible(False)
#
# # Adjust layout
# fig_acc.suptitle('ACC Scores for Different Datasets', fontsize=16)
# fig_l2norm.suptitle('l2-norm Scores for Different Datasets', fontsize=16)
#
# fig_acc.savefig('./ACCAblation.png', bbox_inches='tight')
# fig_l2norm.savefig('./l2-normAblation.png', bbox_inches='tight')
#
# plt.show()
