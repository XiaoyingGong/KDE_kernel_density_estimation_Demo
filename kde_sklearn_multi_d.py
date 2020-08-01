# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/31 16:30  
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

class KDE:
    def __init__(self, data):
        self.data = data

    def fit_data(self):
        h = self.h_determination()
        self.kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(self.data)
        return self

    def get_pdf(self, x):
        pdf = np.exp(self.kde.score_samples(x))
        return pdf

    def get_rank_pdf(self, percentage):
        pdf = np.exp(self.kde.score_samples(self.data))
        x_data_len = len(pdf)
        pre_num = int(np.floor(x_data_len * (1-percentage)))
        sorted_index = np.argsort(pdf)
        self.pre_sorted_index = sorted_index[pre_num:]
        pre_sorted_pdf = pdf[self.pre_sorted_index]
        return self.pre_sorted_index[::-1], pre_sorted_pdf[::-1]

    def h_determination(self):
        _, dim = self.data.shape
        h = np.zeros(dim)
        data_len = len(self.data)
        for i in range(dim):
            h[i] = 1.05 * np.std(self.data[:, i]) * data_len ** (-1 / 6)
        return np.mean(h)
