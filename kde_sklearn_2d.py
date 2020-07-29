# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/28 9:39  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
import time
import generate_data




def h_determination(data):
    _, dim = data.shape
    h = np.zeros(dim)
    data_len = len(data)
    for i in range(dim):
        h[i] = 1.05 * np.std(data[:, i]) * data_len ** (-1 /6)
    return h

x, y = np.mgrid[-10:10:0.5, -10:10:0.5]
x = x.flatten().reshape(-1, 1)
y = y.flatten().reshape(-1, 1)
x_plot = np.hstack((x, y))

print("拟合的数据量为:", len(x_plot))

X = generate_data.mixture_normal_2d(10000)
h = np.mean(h_determination(X))
print(h)
time_1 = time.time()
kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X)
log_dens = kde.score_samples(x_plot)  # 返回的是点x对应概率的log值，要使用exp求指数还原。
dens = np.exp(log_dens)
time_2 = time.time()
print(time_2 - time_1)

plt.hist2d(X[:, 0], X[:, 1], bins=40, density=True)
# plt.plot(X_plot, np.exp(log_dens))
result = dens.reshape([40, 40])
x, y = np.mgrid[-10:10:0.5, -10:10:0.5]
plt.contour(x, y, result, cmap='gray_r')

plt.show()