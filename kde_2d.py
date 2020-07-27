# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/27 21:11  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np

def gauss_kernel(x):
    K = (1 / (np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2))
    return K

def get_kde_2d(x, data, h, kernel_fun):
    data_N, dim = data.shape
    x_N = len(x)
    K = np.ones(x_N)
    for i in range(len(h)):
        x_input = x[:, i].reshape([-1, 1])
        x_input = np.tile(x_input, [1, data_N])
        x_input = (x_input - np.tile(data[:, i], [x_N, 1])) / h[i]
        K *= (1/(data_N * h[i])) * np.sum(kernel_fun(x_input), axis=1)
    return K

def h_determination(data):
    _, dim = data.shape
    h = np.zeros(dim)
    data_len = len(data)
    for i in range(dim):
        h[i] = 1.05 * np.std(data[:, i]) * data_len ** (-1 / 5)
    return h
