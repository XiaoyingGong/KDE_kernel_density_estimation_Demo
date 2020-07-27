# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/27 16:42  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
import kde_2d

def gaussian_2d():
    mus = np.array([2, 2])
    sigmas = np.array([[2, 0], [0, 2]])
    return np.random.multivariate_normal(mus, sigmas, 10000)

def mixture_normal():
    mus_1 = np.array([0, 0])
    sigmas_1 = np.array([[1, 0], [0, 1]])
    mus_2 = np.array([2, 2])
    sigmas_2 = np.array([[1, 0], [0, 1]])

    data_1 = np.random.multivariate_normal(mus_1, sigmas_1, 100000)
    data_2 = np.random.multivariate_normal(mus_2, sigmas_2, 100000)
    data = np.vstack((data_1, data_2))
    return data


def mixture_normal_2():
    mus_1 = np.array([-2, -2])
    sigmas_1 = np.array([[1, 0], [0, 1]])
    mus_2 = np.array([3, 3])
    sigmas_2 = np.array([[1, 0], [0, 1]])

    data_1 = np.random.multivariate_normal(mus_1, sigmas_1, 10000)
    data_2 = np.random.multivariate_normal(mus_2, sigmas_2, 10000)
    data = np.vstack((data_1, data_2))
    return data

data = mixture_normal_2()

x, y = np.mgrid[-8:8:0.5, -8:8:0.5]
print(x.shape)
x = x.flatten().reshape(-1, 1)
y = y.flatten().reshape(-1, 1)
x = np.hstack((x, y))

h = kde_2d.h_determination(data)
h = np.array([0.01, 0.01])
print(h)
result = kde_2d.get_kde_2d(x, data, h, kde_2d.gauss_kernel)
result = result.reshape([32, 32])
plt.hist2d(x=data[:, 0], y=data[:, 1], bins=50)
x, y = np.mgrid[-8:8:0.5, -8:8:0.5]
plt.contour(x, y, result, cmap='gray_r')
plt.show()