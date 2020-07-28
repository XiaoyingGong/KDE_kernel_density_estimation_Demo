# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/28 9:08  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import time

def h_determination(data):
    h = 1.05 * np.std(data) * len(data) ** (-1 / 5)
    return h

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]  # 创建等差数列，在-5和10之间取1000个数

np.random.seed(1)
N = 2000
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
h = h_determination(X.flatten())

time_1 = time.time()
kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X)
log_dens = kde.score_samples(X_plot)  # 返回的是点x对应概率的log值，要使用exp求指数还原。
time_2 = time.time()
print(time_2 - time_1)

plt.hist(X, bins=40,density=True)
plt.plot(X_plot, np.exp(log_dens))
plt.show()