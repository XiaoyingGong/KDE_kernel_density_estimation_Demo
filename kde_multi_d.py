# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/28 17:24  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
class KDE:
    def __init__(self, data):
        self.data = data
        self.N, self.dim = self.data.shape

    def fit_data(self):
        self.data = self.data.T
        self.kernel = stats.gaussian_kde(self.data)
        return self

    def get_pdf(self, x):
        pdf = self.kernel.pdf(x)
        return pdf

    def draw_result(self):
        if self.dim == 1:
            min = np.min(self.data)
            max = np.max(self.data)
            x = np.linspace(min, max, 1000).reshape(-1, 1)
            y = self.kernel(x)
            plt.hist(self.data, bins=40)
            plt.plot(x, y)
            plt.show()
        elif self.dim == 2:
            min_1 = np.min(self.data[0, :])
            max_1 = np.max(self.data[0, :])
            min_2 = np.min(self.data[1, :])
            max_2 = np.min(self.data[1, :])
            x, y = np.mgrid[min_1:max_1:100j, min_2:max_2:100j]
            x_flatten = x.flatten()
            y_flatten = y.flatten()
            input = np.vstack((x_flatten, y_flatten))
            Z = self.kernel(input)
            print("Z:", Z)
            Z = Z.reshape(x.shape)
            plt.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[min_1, max_1, min_2, max_2])
            plt.scatter(self.data[0, :], self.data[1, :], c='black')
            plt.show()
        else:
            raise ValueError("维度为%s的数据不支持绘制" % self.d)


