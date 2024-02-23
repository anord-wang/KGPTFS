import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import seaborn as sns


# RGB格式颜色转换为16进制颜色格式
def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    #print(color)
    return color

# RGB格式颜色转换为16进制颜色格式
def RGB_list_to_Hex(RGB):
    # RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    #print(color)
    return color

# 16进制颜色格式颜色转换为RGB格式
def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    #print(rgb)
    return rgb, [r, g, b]

# 生成渐变色
def gradient_color(color_list, color_sum=100):
    color_center_count = len(color_list)
    # if color_center_count == 2:
    #     color_center_count = 1
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # 生成中间渐变色
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map

#input_colors = ["#40FAFF", "#00EBEB", "#00EB00", "#FFC800", "#FC9600", "#FA0000", "#C800FA", "#FF64FF"]
#input_colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#99004c", "#7e0023"]
#input_colors = ["#008080",'#FFFFFF']
#input_colors = ["#2c6fbb",'#FFFFFF']
#input_colors = ['#0000CD','#FFFFFF']
# input_colors = ["#F06449",'#FDF0DA']    # Orange
# input_colors = ["#1B6A9B",'#D2D4E2']     # Blue
input_colors = ["#FFFFFF","#2F7F52"]      #Green
colors = gradient_color(input_colors)

# sns.palplot(colors)
#main_kl
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [6.63]
# y1 = [6.86]
# y2 = [6.72]
# y3 = [6.81]
# y4 = [7.1]
# y5 = [7.01]
# y6 = [7.43]
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUF framework", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[17], align="center", label="LUCGAN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[34], align="center", label=" CLUVAE", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[51], align="center", label="CVAE", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[68], align="center", label="CGAN", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,1)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("")
# plt.ylim(6,8)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./ablation_metr.pdf')
# plt.show()

# main_JS
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = []
# y1 = [6.86]
# y2 = [6.72]
# y3 = [6.81]
# y4 = [7.1]
# y5 = [7.01]
# y6 = [7.43]
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUPF", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[17], align="center", label="LUCGAN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[34], align="center", label=" CLUVAE", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[51], align="center", label="CVAE", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[68], align="center", label="CGAN", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,1)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_JS")
# plt.ylim(6,8)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./main_js.pdf')
# plt.show()

# main_HD
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [1.81]
# y1 = [6.87]
# y2 = [3.2]
# y3 = [8.1]
# y4 = [7.41]
# y5 = [5.78]
# y6 = [4.9]
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUF framework", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[17], align="center", label="LUCGAN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[34], align="center", label=" CLUVAE", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[51], align="center", label="CVAE", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[68], align="center", label="CGAN", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,1)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_HD")
# plt.ylim(1,11)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./main_hd.pdf')
# plt.show()


#main_wd
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [2.41]
# y1 = [3.27]
# y2 = [3.11]
# y3 = [4.12]
# y4 = [4.23]
# y5 = [3.56]
# y6 = [2.89]
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUF framework", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[17], align="center", label="LUCGAN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[34], align="center", label=" CLUVAE", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[51], align="center", label="CVAE", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[68], align="center", label="CGAN", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,1)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_WD(10^-5)")
# plt.ylim(1,6)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./main_wd.pdf')
# plt.show()

#ab_kl
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [6.63]
# y1 = [6.80]
# y2 = [6.71]
# y3 = [6.85]
# y4 = [7.01]
#
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUF framework", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[25], align="center", label="DSUF-A", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[50], align="center", label=" DSUF-C", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[75], align="center", label="DSUF-ZP", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[99], align="center", label="DSUF-LP", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# # plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,0.7)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_KL")
# plt.ylim(6,7.5)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':8},  ncol=3)
# plt.savefig('./ab_kl.pdf')
# plt.show()



#ab_js
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [6.63]
# y1 = [6.80]
# y2 = [6.71]
# y3 = [6.85]
# y4 = [7.01]
#
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUP Flows", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[25], align="center", label="DSUPF-A", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[50], align="center", label=" DSUPF-C", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[75], align="center", label="DSUPF-ZP", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[99], align="center", label="DSUPF-LP", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# # plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,0.7)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_KL")
# plt.ylim(6,7.5)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./ab_js.pdf')
# plt.show()


#ab_hd
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [1.81]
# y1 = [1.99]
# y2 = [1.87]
# y3 = [2.42]
# y4 = [1.92]
#
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUF framework", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[25], align="center", label="DSUF-A", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[50], align="center", label=" DSUF-C", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[75], align="center", label="DSUF-ZP", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[99], align="center", label="DSUF-LP", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# # plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,0.7)
# plt.ylabel("AVG_HD")
# plt.ylim(1.6,2.7)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':8},  ncol=3)
# plt.savefig('./ab_hd.pdf')
# plt.show()


#ab_wd
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [2.41]
# y1 = [2.59]
# y2 = [2.52]
# y3 = [2.50]
# y4 = [2.56]
#
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUF framework", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[25], align="center", label="DSUF-A", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[50], align="center", label=" DSUF-C", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[75], align="center", label="DSUF-ZP", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[99], align="center", label="DSUF-LP", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# # plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,0.7)
# plt.ylabel("AVG_WD(10^-5)")
# plt.ylim(2.3,2.7)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':8},  ncol=3)
# plt.savefig('./ab_wd.pdf')
# plt.show()

#ro_kl
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [5.21]
# y1 = [5.57]
# y2 = [5.98]
# y3 = [6.63]
#
# bar_width = 0.13
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="10", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[33], align="center", label="25", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[66], align="center", label=" 50", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[99], align="center", label="200", edgecolor = 'black')
#
# plt.xlim(-0.1,0.55)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_KL")
# plt.ylim(5,7.2)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./ro_kl.pdf')
# plt.show()

#ro_js
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [2.41]
# y1 = []
# y2 = [3.11]
# y3 = []
# y4 = []
# y5 = []
# y6 = []
#
# bar_width = 0.13
# x_data = ['DSUPF','LUCGAN','CLUVAE','CVAE','CGAN','DCGAN','WGAN']
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="DSUPF", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[17], align="center", label="LUCGAN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[34], align="center", label=" CLUVAE", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[51], align="center", label="CVAE", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[68], align="center", label="CGAN", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[85], align="center", label="DCGAN", edgecolor = 'black')
# plt.bar(x+6*bar_width+0.12, y6, bar_width, color=colors[99], align="center", label="WGAN", edgecolor = 'black')
#
# plt.xlim(-0.1,1)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_WD(10^-5)")
# plt.ylim(1,11)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./ro_js.pdf')
# plt.show()


#ro_hd
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [1.43]
# y1 = [1.47]
# y2 = [1.52]
# y3 = [1.81]
#
# bar_width = 0.13
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="10", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[33], align="center", label="25", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[66], align="center", label=" 50", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[99], align="center", label="200", edgecolor = 'black')
#
# plt.xlim(-0.1,0.55)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_HD")
# plt.ylim(1,2)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./ro_hd.pdf')
# plt.show()


#ro_wd
# fig = plt.figure(figsize=(5,3.6))
# x = np.arange(1)
# y = [1.51]
# y1 = [1.92]
# y2 = [5.03]
# y3 = [2.41]
#
# bar_width = 0.13
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="10", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[33], align="center", label="25", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[66], align="center", label=" 50", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[99], align="center", label="200", edgecolor = 'black')
#
# plt.xlim(-0.1,0.55)
# # plt.xlabel("Gender", "sb")
# plt.ylabel("AVG_WD(10^-5)")
# plt.ylim(1,6)
# plt.xticks([])
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./ro_wd.pdf')
# plt.show()

#sen_kl
# fig = plt.figure(figsize=(10,8))
# tick_label = ["2","3", "4", "5", "6"]
# x = [1,2,3,4,5]
# mae = [ 6.9, 6.63, 7.12, 6.72, 7.3]
# # mse = [ 0.1468, 0.2187, 0.2148, 0.2551, 0.2994, 0.2818]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7.3,5.3))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mae, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_KL",fontsize = 20)
# plt.ylim(6,8)
# plt.xticks(x, tick_label, fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("Number of Blocks",fontsize = 20)
# # plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
# plt.savefig('./sen_kl.pdf')
# plt.show()


#sen_js
# tick_label = ["2","3", "4", "5", "6"]
# x = [1,2,3,4,5]
# mae = [ 0.1468, 6.87, 6.63, 6.67, 6.83, 7.5]
# # mse = [ 0.1468, 0.2187, 0.2148, 0.2551, 0.2994, 0.2818]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_KL")
# plt.ylim(0,0.4)
# plt.xticks(x, tick_label)
# plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
# plt.savefig('./sen_js.pdf')
# plt.show()


#sen_hd
# tick_label = ["2","3", "4", "5", "6"]
# x = [1,2,3,4,5]
# mae = [ 2.4, 1.81, 2.24, 1.9, 2.37]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7.3,5.3))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mae, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_HD", fontsize = 20)
# plt.ylim(1.5, 2.5)
# plt.xticks(x, tick_label, fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("Number of Blocks",fontsize = 20)
# # plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
#
# plt.savefig('./sen_hd.pdf')
# plt.show()


#sen_wd
# tick_label = ["2","3", "4", "5", "6"]
# x = [1,2,3,4,5]
# mae = [ 3.47, 2.41, 3.72, 4.12, 4.53]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7.3,5.3))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mae, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_WD(10^-5)", fontsize = 20)
# plt.ylim(2,4.7)
# plt.xticks(x, tick_label, fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("Number of Blocks",fontsize = 20)
# # plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
# plt.savefig('./sen_wd.pdf')
# plt.show()




















#sen_kl
# tick_label = ["1","2", "4", "5", "10"]
# x = [1,2,3,4,5]
# mae = [ 7.32, 6.76, 6.63, 6.92, 6.83]
# # mse = [ 0.1468, 0.2187, 0.2148, 0.2551, 0.2994, 0.2818]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7.3,5.3))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mae, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_KL", fontsize = 20)
# plt.ylim(6.5,7.5)
# plt.xticks(x, tick_label, fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("Number of Blocks",fontsize = 20)
# # plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
# plt.savefig('./sen_kl_2.pdf')
# plt.show()


#sen_js
# tick_label = ["2","3", "4", "5", "6"]
# x = [1,2,3,4,5]
# mae = [ 0.1468, 6.87, 6.63, 6.67, 6.83, 7.5]
# # mse = [ 0.1468, 0.2187, 0.2148, 0.2551, 0.2994, 0.2818]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_KL")
# plt.ylim(0,0.4)
# plt.xticks(x, tick_label)
# plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
# plt.savefig('./sen_js.pdf')
# plt.show()


# #sen_hd
# tick_label = ["1","2", "4", "5", "10"]
# x = [1,2,3,4,5]
# mae = [ 3.42, 1.98, 1.81, 2.07, 2.01]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7.3,5.3))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mae, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_HD",fontsize = 20)
# plt.ylim(1.7, 3.5)
# plt.xticks(x, tick_label, fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("Number of Blocks",fontsize = 20)
# # plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
#
# plt.savefig('./sen_hd_2.pdf')
# plt.show()


#sen_wd
# tick_label = ["1","2", "4", "5", "10"]
# x = [1,2,3,4,5]
# mae = [ 4.51, 2.52, 2.41, 3.29, 2.98]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7.3,5.3))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mae, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("AVG_WD(10^-5)", fontsize = 20)
# plt.ylim(2.2,4.7)
# plt.xticks(x, tick_label, fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel("Number of Blocks",fontsize = 20)
# # plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
# plt.savefig('./sen_wd_2.pdf')
# plt.show()










# Ab_tran
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(2)
# y = [22.55,41.79]
# y1 = [23.47,43.16]
# y2 = [23.37,42.98]
# y3 = [23.93, 43.12]
# y4 = [22.84,42.46]
# y5 = [23.12,42.73]
#
# bar_width = 0.13
# tick_label = ["MAE", "RMSE"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="SAUP", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[20], align="center", label="SAUP-A", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[40], align="center", label="SAUP-NT", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# plt.xlim(-0.15,1.9)
# plt.xticks(x+2.5*bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(16,51)
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./Ab_tran.pdf')
# plt.show()


#Ab_dl
# fig = plt.figure(figsize=(4.3,3))
# x = np.arange(2)
# y = [23.22,41.51]
# y1 = [23.6500,42.8400]
# y2 = [23.3000,42.0300]
# y3 = [30.16,51.36]
# y4 = [24.11,42.7]
#
# bar_width = 0.15
# tick_label = ["MAE", "RMSE"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="SAUP")
# plt.bar(x+bar_width, y1, bar_width, color=colors[20], align="center", label="SAUP-A")
# plt.bar(x+2*bar_width, y2, bar_width, color=colors[40], align="center", label="SAUP-N")
# plt.bar(x+3*bar_width, y3, bar_width, color=colors[60], align="center", label="SAUP-G")
# plt.bar(x+4*bar_width, y4, bar_width, color=colors[80], align="center", label="SAUP-C")
#
# #plt.xlabel("Gender")
# # plt.ylabel()
# # plt.xlim(0.2,2.8)
# plt.xticks(x+2*bar_width, tick_label)
# plt.ylim(16,60)
# plt.legend(loc="upper right", prop= {'size':8},  ncol=3)
# plt.savefig('./Ab_dl.pdf')
# plt.show()



#hy_mtgnn
# fig = plt.figure(figsize=(4.5,4.1))
# x = np.arange(5)
# y = [9.65, 7.39, 6.45, 6.08, 6.71]
# y1 = [6.97, 5.57, 4.82, 4.42, 5.13]
# # y2 = [6.44, 1.74]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.35
# tick_label = ["2", "3", "4", "5", "6"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[99], label="MAE", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="RMSE", edgecolor = 'none')
# # plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width-0.16, tick_label, fontsize = 10)
# plt.ylim(4,10)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.yticks(fontsize = 10)
# plt.xlabel("Number of Coupling Layers", fontsize = 12)
# plt.legend(loc="upper right", prop= {'size':20},  ncol=1)
# plt.savefig('./hy_mtgnn.pdf')
# plt.show()



#hy_astgcn
# fig = plt.figure(figsize=(4.5,4.1))
# x = np.arange(5)
# y = [5.23, 3.56, 2.2, 1.65, 1.98]
# y1 = [6.15, 4.44, 2.74, 2.48, 2.56]
# # y2 = [6.44, 1.74]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.35
# tick_label = ["2", "3", "4", "5", "6"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[99], label="MAE", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="RMSE", edgecolor = 'none')
# # plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width-0.16, tick_label, fontsize = 10)
# plt.ylim(1.5,7)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.yticks(fontsize = 10)
# plt.xlabel("Number of Coupling Layers", fontsize = 12)
# plt.legend(loc="upper right", prop= {'size':20},  ncol=1)
# plt.savefig('./hy_astgcn.pdf')
# plt.show()

#hy_tgcn
# fig = plt.figure(figsize=(4.5,4.1))
# x = np.arange(5)
# y = [5.92, 5.27, 4.89, 4.74, 5.09]
# y1 = [6.29, 5.68, 5.35, 5.1, 5.43]
# # y2 = [6.44, 1.74]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.35
# tick_label = ["2", "3", "4", "5", "6"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[99], label="MAE", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="RMSE", edgecolor = 'none')
# # plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width-0.16, tick_label, fontsize = 10)
# plt.ylim(2,9)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.yticks(fontsize = 10)
# plt.xlabel("Number of Coupling Layers", fontsize = 12)
# plt.legend(loc="upper right", prop= {'size':20},  ncol=1)
# plt.savefig('./hy_tgcn.pdf')
# plt.show()


#hy_1_solar_mtgnn
# fig = plt.figure(figsize=(4.5,4.1))
# x = np.arange(5)
# y = [29.4, 26.87, 26.2, 26.57, 28.76]
# y1 = [27.22, 23.67, 22.9, 23.41, 26.83]
# # y2 = [6.44, 1.74]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.35
# tick_label = ["2", "3", "4", "5", "6"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[99], label="MAE", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="RMSE", edgecolor = 'none')
# # plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width-0.16, tick_label, fontsize = 10)
# plt.ylim(22,33)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.yticks(fontsize = 10)
# plt.xlabel("Number of Coupling Layers", fontsize = 12)
# plt.legend(loc="upper right", prop= {'size':20},  ncol=1)
# plt.savefig('./hy_1_solar_mtgnn.pdf')
# plt.show()


#hy_1_solar_astgcn
# fig = plt.figure(figsize=(4.5,4.1))
# x = np.arange(5)
# y = [9.45, 8.13, 7.48, 8.41, 9.21]
# y1 = [9.13, 7.68, 6.87, 8.14, 9.02]
# # y2 = [6.44, 1.74]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.35
# tick_label = ["2", "3", "4", "5", "6"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[99], label="MAE", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="RMSE", edgecolor = 'none')
# # plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width-0.16, tick_label, fontsize = 10)
# plt.ylim(6,11)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.yticks(fontsize = 10)
# plt.xlabel("Number of Coupling Layers", fontsize = 12)
# plt.legend(loc="upper right", prop= {'size':20},  ncol=1)
# plt.savefig('./hy_1_solar_astgcn.pdf')
# plt.show()

#hy_1_solar_tgcn
# fig = plt.figure(figsize=(4.5,4.1))
# x = np.arange(5)
# y = [7.12, 6.54, 6.32, 6.43, 7.35]
# y1 = [4.98, 4.45, 4.19, 4.31, 5.06]
# # y2 = [6.44, 1.74]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.35
# tick_label = ["2", "3", "4", "5", "6"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[99], label="MAE", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="RMSE", edgecolor = 'none')
# # plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width-0.16, tick_label, fontsize = 10)
# plt.ylim(4,10)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.yticks(fontsize = 10)
# plt.xlabel("Number of Coupling Layers", fontsize = 12)
# plt.legend(loc="upper right", prop= {'size':20},  ncol=1)
# plt.savefig('./hy_1_solar_tgcn.pdf')
# plt.show()







#Ab_metr
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(3)
# y = [6.83, 1.89, 5.49]
# y1 = [6.08, 1.65, 4.74]
# y2 = [6.44, 1.74, 4.93]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.25
# tick_label = ["MTGNN", "ASTGCN", "T-GCN"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="TRF w/o BZ", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="TRF", edgecolor = 'none')
# plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(1,8)
# plt.tick_params(labelsize = 12)
# plt.ylabel("Recovering Gap of MAE", fontsize = 15)
# plt.legend(loc="upper right", prop= {'size':12},  ncol=1)
# plt.savefig('./Ab_metr.pdf')
# plt.show()



# hori_metr
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(3)
# y = [9.6, 10.04, 6.08]
# y1 = [1.29, 1.47, 1.65]
# y2 = [3.83,4.26, 4.74]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.25
# tick_label = ["Horizon 3", "Horizon 6", "Horizon 12"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[10], label="MTGNN", edgecolor = 'black')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="ASTGCN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="T-GCN", edgecolor = 'black')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(1,11)
# plt.yticks(fontsize = 12)
# plt.ylabel("Recovering Gap of MAE", fontsize = 13)
# plt.legend(loc="upper right", prop= {'size':12},  ncol=1)
# plt.savefig('./hori_metr.pdf')
# plt.show()

# hori_solar
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(3)
# y = [23.55, 30.31, 26.17]
# y1 = [5.09, 6.74, 7.48]
# y2 = [2.77,4.74, 6.32]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.25
# tick_label = ["Horizon 3", "Horizon 6", "Horizon 12"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[10], label="MTGNN", edgecolor = 'black')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="ASTGCN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="T-GCN", edgecolor = 'black')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(1,38)
# plt.yticks(fontsize = 12)
# plt.ylabel("Recovering Gap of MAE", fontsize = 13)
# plt.legend(loc="upper right", prop= {'size':12},  ncol=1)
# plt.savefig('./hori_solar.pdf')
# plt.show()


# hori_traffic
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(3)
# y = [1.14, 1.74, 1.86]
# y1 = [0.12, 0.16, 0.17]
# y2 = [0.2,0.45, 0.65]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.25
# tick_label = ["Horizon 3", "Horizon 6", "Horizon 12"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[10], label="MTGNN", edgecolor = 'black')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="ASTGCN", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="T-GCN", edgecolor = 'black')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(0,3)
# plt.yticks(fontsize = 9)
# plt.ylabel("Recovering Gap of MAE", fontsize = 13)
# plt.legend(loc="upper right", prop= {'size':12},  ncol=1)
# plt.savefig('./hori_traffic.pdf')
# plt.show()







# ab_traffic
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(3)
# y = [2.23, 0.82, 1.86]
# y1 = [1.86, 0.17, 0.65]
# y2 = [2.01, 0.6, 1.24]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.25
# tick_label = ["MTGNN", "ASTGCN", "T-GCN"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="TRF w/o BZ", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="TRF", edgecolor = 'none')
# plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(0,3)
# plt.yticks(fontsize = 9)
# plt.ylabel("Recovering Gap of MAE", fontsize = 15)
# plt.legend(loc="upper right", prop= {'size':12},  ncol=1)
# plt.savefig('./Ab_traffic.pdf')
# plt.show()



# ab_solar
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(3)
# y = [26.97, 8.69, 7.08]
# y1 = [26.17, 7.48, 6.32]
# y2 = [26.49, 7.76, 6.91]
# # y3 = [22.56, 41.79]
# # y4 = [22.62, 41.23]
# # y5 = [23.65, 43.50]
#
# bar_width = 0.25
# tick_label = ["MTGNN", "ASTGCN", "T-GCN"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="TRF w/o BZ", edgecolor = 'none')
# plt.bar(x+bar_width+0.05, y1, bar_width, color=colors[40], align="center", label="TRF", edgecolor = 'none')
# plt.bar(x+2*bar_width+0.1, y2, bar_width, color=colors[99], align="center", label="TRF w/o VAML", edgecolor = 'none')
# # plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# # plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# # plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# # plt.xlim(-0.15,1.9)
# plt.xticks(x+bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(6,28)
# plt.tick_params(labelsize = 12)
# plt.ylabel("Recovering Gap of MAE", fontsize = 15)
# plt.legend(loc="upper right", prop= {'size':12},  ncol=1)
# plt.savefig('./Ab_solar.pdf')
# plt.show()




#Ab_nb
# fig = plt.figure(figsize=(4.5,3.3))
# x = np.arange(2)
# y = [21.80, 40.1700]
# y1 = [22.56, 40.77]
# y2 = [22.69, 41.38]
# y3 = [22.56, 41.79]
# y4 = [22.62, 41.23]
# y5 = [23.65, 43.50]
#
# bar_width = 0.13
# tick_label = ["MAE", "RMSE"]
#
# plt.bar(x, y, bar_width, align="center", color=colors[0], label="SAUP", edgecolor = 'black')
# plt.bar(x+bar_width+ 0.02, y1, bar_width, color=colors[20], align="center", label="SAUP-A", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[40], align="center", label="SAUP-NT", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[60], align="center", label="SAUP-NS", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[80], align="center", label="SAUP-G", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.1, y5, bar_width, color=colors[99], align="center", label="SAUP-C", edgecolor = 'black')
#
#
# plt.xlim(-0.15,1.9)
# plt.xticks(x+2.5*bar_width+0.05, tick_label, fontsize = 12)
# plt.ylim(16,52)
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./Ab_nb.pdf')
# plt.show()


# Number of hidden layers in each MADE mtgnn metr
# tick_label = ["1", "2","3", "4", "5", "6"]
# x = [1,2,3,4,5,6]
# mse = [ 6.08, 6.23, 6.56, 7.24, 8.01, 9.13]
# mae = [4.42, 4.57, 4.89, 5.42, 6.19, 6.97]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.ylim(3,10)
# plt.yticks(fontsize = 10)
# plt.xticks(x, tick_label, fontsize = 10)
# plt.legend(loc="lower right", prop= {'size':20})
# plt.xlabel("Number of Hidden Layers",fontsize = 12)
# plt.grid(which="major")
# plt.savefig('./hyper_2_metr_mtgnn.pdf')
# plt.show()



# Number of hidden layers in each MADE astgcn metr
# tick_label = ["1", "2","3", "4", "5", "6"]
# x = [1,2,3,4,5,6]
# mse = [ 1.65, 1.89, 2.23, 3.98, 5.91, 7.23]
# mae = [2.48, 2.79, 3.52, 4.61, 6.89, 9.76]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.ylim(1,10)
# plt.yticks(fontsize = 10)
# plt.xticks(x, tick_label, fontsize = 10)
# plt.legend(loc="lower right", prop= {'size':20})
# plt.xlabel("Number of Hidden Layers",fontsize = 12)
# plt.grid(which="major")
# plt.savefig('./hyper_2_metr_astgcn.pdf')
# plt.show()


# Number of hidden layers in each MADE tgcn metr
# tick_label = ["1", "2","3", "4", "5", "6"]
# x = [1,2,3,4,5,6]
# mse = [ 4.74, 5.03, 5.78, 7.38, 9.16, 11.89]
# mae = [5.1, 5.67, 6.81, 8.03, 10.2, 12.98]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.ylim(4,14)
# plt.yticks(fontsize = 10)
# plt.xticks(x, tick_label, fontsize = 10)
# plt.legend(loc="lower right", prop= {'size':20})
# plt.xlabel("Number of Hidden Layers",fontsize = 12)
# plt.grid(which="major")
# plt.savefig('./hyper_2_metr_tgcn.pdf')
# plt.show()


# Number of hidden layers in each MADE mtgnn solar
# tick_label = ["1", "2","3", "4", "5", "6"]
# x = [1,2,3,4,5,6]
# mse = [26.2, 27.81, 29.03, 33.25, 38.98, 43.16]
# mae = [22.9, 23.67, 25.08, 29.17, 34.98, 40.42]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.ylim(20,45)
# plt.yticks(fontsize = 10)
# plt.xticks(x, tick_label, fontsize = 10)
# plt.legend(loc="lower right", prop= {'size':20})
# plt.xlabel("Number of Hidden Layers",fontsize = 12)
# plt.grid(which="major")
# plt.savefig('./hyper_2_solar_mtgnn.pdf')
# plt.show()


# Number of hidden layers in each MADE astgcn solar
# tick_label = ["1", "2","3", "4", "5", "6"]
# x = [1,2,3,4,5,6]
# mse = [7.48, 8.02, 8.97, 10.34, 13.58, 17.92]
# mae = [6.87, 7.03, 8.21, 10.78, 13.97, 18.06]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("Recovering Gap", fontsize = 12)
# plt.ylim(6,19)
# plt.yticks(fontsize = 10)
# plt.xticks(x, tick_label, fontsize = 10)
# plt.legend(loc="lower right", prop= {'size':20})
# plt.xlabel("Number of Hidden Layers",fontsize = 12)
# plt.grid(which="major")
# plt.savefig('./hyper_2_solar_astgcn.pdf')
# plt.show()


# Number of hidden layers in each MADE tgcn solar
tick_label = ["1", "2","3", "4", "5", "6"]
x = [1,2,3,4,5,6]
mse = [6.32, 6.97, 8.23, 9.99, 12.87, 18.21]
mae = [4.19, 4.56, 5.97, 7.34, 10.23, 15.76]
x_major_locator = MultipleLocator(1)
y_major_locator=MultipleLocator(0.1)
fig = plt.figure(figsize=(7,5))
ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=7)
plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=7)
# plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
plt.ylabel("Recovering Gap", fontsize = 12)
plt.ylim(4,19)
plt.yticks(fontsize = 10)
plt.xticks(x, tick_label, fontsize = 10)
plt.legend(loc="lower right", prop= {'size':20})
plt.xlabel("Number of Hidden Layers",fontsize = 12)
plt.grid(which="major")
plt.savefig('./hyper_2_solar_tgcn.pdf')
plt.show()






#Time
# tick_label = ["Nbeats", "Dlinear","Transformer", "Informer", "AGCRN", "ASTGCN"]
# x = [1,2,3,4,5,6]
# mse = [ 0.1468, 0.2187, 0.2148, 0.2551, 0.2994, 0.2818]
# mae = [0.0104, 0.0676, 0.1143, 0.1243, 0.1901, 0.1658]
# # con = [0.7565,0.7634,0.7754,0.7896,0.7886]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='backbones only',markersize=7)
# plt.plot(x, mse, marker='o',c='#965454',label='whole framework',markersize=7)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# plt.ylabel("Average Training Time per epoch(min)")
# plt.ylim(0,0.4)
# plt.xticks(x, tick_label)
# plt.legend(loc="upper right", prop= {'size':10})
# plt.grid(which="major")
# plt.savefig('./time.pdf')
# plt.show()


#ks
# tick_label = [1,2,3,4,5]
# x = [1,2,3,4,5]
# mse = [42.66,41.21,43.93,41.7,41.89]
# mae = [23.76,22.39,24.67,22.72,23.11]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=9)
# plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=9)
# # plt.plot(x, mse, marker='o',c='#965454',label='MSE',markersize=9)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# # plt.xlabel("Kernel Size of GCN layers")
# plt.ylim(20,50)
# plt.xticks(x, tick_label)
# plt.legend(loc="center", prop= {'size':20})
# plt.grid(which="major")
# plt.savefig('./ks_sen.pdf')
# plt.show()
#
#
# #hidden
# tick_label = [25, 50,64,100,128]
# x = [1,2,3, 4, 5]
# mse = [41.4900, 41.21, 42.58, 45.77, 49.08]
# mae = [22.6600, 22.39, 23.84, 25.14, 29.15]
# # con = [0.7565,0.7634,0.7754,0.7896,0.7886]
# x_major_locator = MultipleLocator(1)
# y_major_locator=MultipleLocator(0.1)
# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# # ax.xaxis.set_major_locator(x_major_locator)
# # ax.yaxis.set_major_locator(y_major_locator)
# plt.plot(x, mae, marker='x',c='#656565',label='MAE',markersize=9)
# plt.plot(x, mse, marker='o',c='#965454',label='RMSE',markersize=9)
# # plt.plot(x, mape, marker='*',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, gd, marker='o',c='b',label='Average Training Time for each epoch',markersize=5)
# # plt.plot(x, con, marker='x',c='purple',label='Consistency',markersize=7)
# # plt.xlabel("hidden size of Normalizing flows")
# plt.ylim(20,50)
# plt.xticks(x, tick_label)
# plt.legend(loc="center", prop= {'size':20})
# plt.grid(which="major")
# plt.savefig('./hidden_sen.pdf')
# plt.show()


#RO_nb.jpg
# # fig = plt.figure(figsize=(4.3,3))
# fig = plt.figure(figsize=(4.5,3.4))
# x = np.arange(2)
#
# # MAE
# # y = [14.81,34.31]
# # y1 = [31.2,41.47]
# # y2 = [22.58,33.66]
# # y3 = [20.86, 24.25]
# # y4 = [25.02,25.77]
# # y5 = [14.52,21.21]
#
# # RMSE
# # y = [23.41,48.06]
# # y1 = [51.54,64.94]
# # y2 = [37.48,52.97]
# # y3 = [30.49, 34.55]
# # y4 = [35.57,50.39]
# # y5 = [23.07,31.47]
#
# # MAPE
# y = [65.64,71.13]
# y1 = [63.57,92.35]
# y2 = [68.87,103.07]
# y3 = [56.25, 70.56]
# y4 = [72.33,95.51]
# y5 = [72.50,116.02]
#
# bar_width = 0.13
# tick_label = ["Horizon 6", "Horizon 12"]
#
# plt.bar(x, y, bar_width, align="center", color='#F06449', label="Period 1", edgecolor = 'black')
# plt.bar(x+bar_width+0.02, y1, bar_width, color=colors[0], align="center", label="Period 2", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[25], align="center", label="Period 3", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[50], align="center", label="Period 4", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[75], align="center", label="Period 5", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.10, y5, bar_width, color=colors[99], align="center", label="Period 6", edgecolor = 'black')
#
# plt.xlim(-0.15,1.9)
# plt.xticks(x+2.5*bar_width+0.05, tick_label, fontsize = 12)
#
# # MAE
# # plt.ylabel('MAE', font= {'size':12})
# # plt.ylim(10,50)
#
# # RMSE
# # plt.ylabel('RMSE', font= {'size':12})
# # plt.ylim(20,78)
#
# # MAPE
# plt.ylabel('MAPE(%)', font= {'size':12})
# plt.ylim(50,135)
#
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# # plt.savefig('./Ro_nb_3_mae.pdf')
# # plt.savefig('./Ro_nb_3_rmse.pdf')
# plt.savefig('./Ro_nb_3_mape.pdf')


# plt.legend(loc="upper right", prop= {'size':8},  ncol=2)
# plt.savefig('./Ro_tran.pdf')
# plt.show()




#RO_dl
# # fig = plt.figure(figsize=(4.3,3))
# fig = plt.figure(figsize=(4.5,3.4))
# x = np.arange(2)
# y = [23.18,27.19]
# y1 = [29.54,40.21]
# y2 = [29.06,29.87]
# y3 = [18.23,22.52]
# y4 = [26.98,38.91]
# y5 = [15.45,20.6]
#
# bar_width = 0.13
# tick_label = ["Horizon 6", "Horizon 12"]
#
# plt.bar(x, y, bar_width, align="center", color='#F06449', label="Time Span 1", edgecolor = 'black')
# plt.bar(x+bar_width+0.02, y1, bar_width, color=colors[0], align="center", label="Time Span 2", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[25], align="center", label="Time Span 3", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[50], align="center", label="Time Span 4", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[75], align="center", label="Time Span 5", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.10, y5, bar_width, color=colors[99], align="center", label="Time Span 6", edgecolor = 'black')
#
# plt.xlim(-0.15,1.9)
# plt.xticks(x+2.5*bar_width+0.05, tick_label)
# plt.ylim(10,55)
# plt.legend(loc="upper right", prop= {'size':8},  ncol=2)
# plt.savefig('./Ro_dl.pdf')
# plt.show()

#Ro_tran
# fig = plt.figure(figsize=(4.5,3.4))
# x = np.arange(2)
# # MAE
# y = [23.67,25.19]
# y1 = [30.3,43.24]
# y2 = [36.27,39.91]
# y3 = [19.34,24.22]
# y4 = [28.11,39.72]
# y5 = [18.46,24.91]
#
# # RMSE
# # y = [34.05,37.95]
# # y1 = [51.47,69.21]
# # y2 = [59.54,65.07]
# # y3 = [27.54,33.63]
# # y4 = [53.39,68.65]
# # y5 = [29.36,37.48]
#
# # MAPE
# # y = [48.24,115.28]
# # y1 = [53.11,73.40]
# # y2 = [76.36,95.73]
# # y3 = [52.98,66.75]
# # y4 = [98.93,165.92]
# # y5 = [81.91,118.85]
#
# bar_width = 0.13
# tick_label = ["Horizon 6", "Horizon 12"]
#
# plt.bar(x, y, bar_width, align="center", color='#F06449', label="Period 1", edgecolor = 'black')
# plt.bar(x+bar_width+0.02, y1, bar_width, color=colors[0], align="center", label="Period 2", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[25], align="center", label="Period 3", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[50], align="center", label="Period 4", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[75], align="center", label="Period 5", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.10, y5, bar_width, color=colors[99], align="center", label="Period 6", edgecolor = 'black')
#
# plt.xlim(-0.15,1.9)
# plt.xticks(x+2.5*bar_width+0.05, tick_label, fontsize = 12)
#
# # MAE
# plt.ylabel('MAE', font= {'size':12})
# plt.ylim(10,55)
#
# # RMSE
# # plt.ylabel('RMSE', font= {'size':12})
# # plt.ylim(25,83)
#
# # MAPE
# # plt.ylabel('MAPE(%)', font= {'size':12})
# # plt.ylim(45,200)
#
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# plt.savefig('./Ro_tran_3_mae.pdf')
# # plt.savefig('./Ro_tran_3_rmse.pdf')
# # plt.savefig('./Ro_tran_3_mape.pdf')
#
#
# # plt.legend(loc="upper right", prop= {'size':8},  ncol=2)
# # plt.savefig('./Ro_tran.pdf')
# plt.show()





# #RO_as.jpg
# # fig = plt.figure(figsize=(4.3,3))
# fig = plt.figure(figsize=(4.5,3.4))
# x = np.arange(2)
#
# # MAE
# # y = [33.18,55.44]
# # y1 = [36.7,48.45]
# # y2 = [34.86,42.85]
# # y3 = [23.37,29.21]
# # y4 = [36.29,45.31]
# # y5 = [18.39,38.72]
#
# # RMSE
# # y = [45.45,73.45]
# # y1 = [57.72,71.52]
# # y2 = [56.27,67.93]
# # y3 = [32.71,39.01]
# # y4 = [62.4,74.47]
# # y5 = [26.96,52]
#
# # MAPE
# y = [80.82,112.87]
# y1 = [77.23,124.31]
# y2 = [90.59,101.61]
# y3 = [70.17,108.55]
# y4 = [143.92,179.10]
# y5 = [103.29,216.90]
#
# bar_width = 0.13
# tick_label = ["Horizon 6", "Horizon 12"]
#
# plt.bar(x, y, bar_width, align="center", color='#F06449', label="Period 1", edgecolor = 'black')
# plt.bar(x+bar_width+0.02, y1, bar_width, color=colors[0], align="center", label="Period 2", edgecolor = 'black')
# plt.bar(x+2*bar_width+0.04, y2, bar_width, color=colors[25], align="center", label="Period 3", edgecolor = 'black')
# plt.bar(x+3*bar_width+0.06, y3, bar_width, color=colors[50], align="center", label="Period 4", edgecolor = 'black')
# plt.bar(x+4*bar_width+0.08, y4, bar_width, color=colors[75], align="center", label="Period 5", edgecolor = 'black')
# plt.bar(x+5*bar_width+0.10, y5, bar_width, color=colors[99], align="center", label="Period 6", edgecolor = 'black')
#
# plt.xlim(-0.15,1.9)
# plt.xticks(x+2.5*bar_width+0.05, tick_label)
#
#
# # MAE
# # plt.ylabel('MAE', font= {'size':12})
# # plt.ylim(10,70)
#
# # RMSE
# # plt.ylabel('RMSE', font= {'size':12})
# # plt.ylim(25,88)
#
# # MAPE
# plt.ylabel('MAPE(%)', font= {'size':12})
# plt.ylim(60,258)
#
# plt.legend(loc="upper right", prop= {'size':9.5},  ncol=3)
# # plt.savefig('./Ro_as_3_mae.pdf')
# # plt.savefig('./Ro_as_3_rmse.pdf')
# plt.savefig('./Ro_as_3_mape.pdf')
#
#
# # plt.legend(loc="upper right", prop= {'size':8},  ncol=2)
# # plt.savefig('./Ro_as.pdf')
# plt.show()








# data = pd.read_excel('/Users/solomon/Documents/research/KDD2023/data/Beijing/Beijing_PM_1.xlsx')
# data = data.interpolate()
# # print(data)
# fig = plt.figure(figsize=(13,9))
# # plt.plot(data['data_1'],c='#FFA500',label='station 1')
# # plt.plot(data['data_2'],c='#965454',label='station 2')
# # plt.plot(data['data_3'],c='#6495ED',label='station 3')
# plt.plot(data['data_4'],c='#FFA500',label='station 1')
# plt.plot(data['data_5'],c='#965454',label='station 2')
# plt.plot(data['data_6'],c='#6495ED',label='station 3')
# plt.xlabel("Time (hour)", fontdict={'size': 35})
# # plt.ylabel("PM2.5 (μg/m^3)", fontdict={'size':35})
# plt.ylabel("PM10 (μg/m^3)", fontdict={'size':35})
# plt.tick_params(labelsize = 35)
# plt.legend(loc="upper right", prop= {'size':25},  ncol=2)
# # plt.savefig('./shift_pm25.pdf')
# plt.savefig('./shift_pm10.pdf')
# plt.show()
