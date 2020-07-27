# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/23 8:56  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import time
def gauss_kernel(x):
    K = (1 / (np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2))
    return K

def get_kde(x, data, h, kernel_fun):
    data_N = len(data)
    x_N = len(x)
    x = x.reshape([-1, 1])
    x = np.tile(x, [1, data_N])
    x = (x - np.tile(data, [x_N, 1])) / h
    K = (1/(data_N * h)) * np.sum(kernel_fun(x), axis=1)
    return K

def h_determination(data):
    h = 1.05 * np.std(data) * len(data) ** (-1 / 5)
    return h