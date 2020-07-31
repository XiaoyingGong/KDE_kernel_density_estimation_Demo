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
        self.kernel = None
        self.pre_sorted_index = None

    def fit_data(self):
        if self.dim == 1:
            self.data = self.data.flatten()
        else:
            self.data = self.data.T
        self.kernel = stats.gaussian_kde(self.data)
        return self

    def get_pdf(self, x):
        if self.dim == 1:
            x = x.flatten()
        else:
            x = x.T
        pdf = self.kernel.pdf(x)
        return pdf

    def get_rank_pdf(self, percentage):
        pdf = self.kernel.pdf(self.data)
        x_data_len = len(pdf)
        pre_num = int(np.floor(x_data_len * (1-percentage)))
        sorted_index = np.argsort(pdf)
        self.pre_sorted_index = sorted_index[pre_num:]
        pre_sorted_pdf = pdf[self.pre_sorted_index]
        return self.pre_sorted_index[::-1], pre_sorted_pdf[::-1]

    def draw_result(self):
        if self.dim == 1:
            plt.figure()
            min_v = np.min(self.data)
            max_v = np.max(self.data)
            x = np.linspace(min_v, max_v, 1000)
            y = self.kernel(x)
            plt.hist(self.data, bins=40, density=True)
            plt.plot(x, y)
        elif self.dim == 2:
            plt.figure()
            min_1 = np.min(self.data[0, :])
            max_1 = np.max(self.data[0, :])
            min_2 = np.min(self.data[1, :])
            max_2 = np.max(self.data[1, :])
            x, y = np.mgrid[min_1:max_1:100j, min_2:max_2:100j]
            x_flatten = x.flatten()
            y_flatten = y.flatten()
            input = np.vstack((x_flatten, y_flatten))
            Z = self.kernel(input)
            Z = Z.reshape(x.shape)
            plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[min_1, max_1, min_2, max_2])
            plt.xlim([min_1, max_1])
            plt.ylim([min_2, max_2])
            plt.scatter(self.data[0, :], self.data[1, :], c='black', s=1)
            if self.pre_sorted_index is not None:
                plt.scatter(self.data[0, self.pre_sorted_index], self.data[1, self.pre_sorted_index], c='r', s=2)
        else:
            raise ValueError("维度为%s的数据不支持绘制" % self.d)


