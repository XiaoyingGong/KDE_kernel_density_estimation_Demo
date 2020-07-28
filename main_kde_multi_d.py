# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/28 20:53  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
from kde_multi_d import KDE
from scipy import stats
np.random.seed(1)
# ----------------------------------- create data 生成数据 -----------------------------------
def standard_normal_1d():
    data = np.random.randn(2000)
    return data.reshape([1, -1])

def mixture_normal_1d():
    data_1 = np.random.normal(0, 1, 2500)
    data_2 = np.random.normal(5, 2, 2500)
    data = np.hstack((data_1, data_2))
    return data

def mixture_normal_2d(num):
    mus_1 = np.array([0, 0])
    sigmas_1 = np.array([[1, 0], [0, 1]])
    mus_2 = np.array([4, 4])
    sigmas_2 = np.array([[1, 0], [0, 1]])
    data_1 = np.random.multivariate_normal(mus_1, sigmas_1, num)
    data_2 = np.random.multivariate_normal(mus_2, sigmas_2, num)
    data = np.vstack((data_1, data_2))
    return data


# data = mixture_normal_2d(2000)
# kde = KDE(data).fit_data()
# kde.draw_result()
# min_1 = np.min(data.T[0, :])
# max_1 = np.max(data.T[0, :])
# min_2 = np.min(data.T[1, :])
# max_2 = np.min(data.T[1, :])
# x, y = np.mgrid[min_1:max_1:100j, min_2:max_2:100j]
# x_flatten = x.flatten()
# y_flatten = y.flatten()
# input = np.vstack((x_flatten, y_flatten))
# kernel = stats.gaussian_kde(data.T)
# print(kernel(input))