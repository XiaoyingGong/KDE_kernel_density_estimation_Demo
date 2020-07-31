# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/28 20:53  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
from kde_multi_d import KDE
from scipy import stats
import matplotlib.pyplot as plt
import generate_data

data_1d = generate_data.mixture_normal_1d(2000).reshape([-1, 1])
kde_1d = KDE(data_1d).fit_data()
kde_1d.draw_result()

data_2d = generate_data.uniform_2d(2000)
kde_2d = KDE(data_2d).fit_data()
kde_2d.get_rank_pdf(0.4)
kde_2d.draw_result()
plt.show()


data_4d = generate_data.mixture_normal_4d(500)
print(data_4d.shape)
kde_4d = KDE(data_4d).fit_data()
pdf = kde_4d.get_pdf(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]))
print(pdf)

data_5d = generate_data.mixture_normal_5d(500)
print(data_5d.shape)
kde_5d = KDE(data_5d).fit_data()
pdf = kde_5d.get_pdf(np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]]))
print(pdf)

data_6d = generate_data.mixture_normal_6d(500)
print(data_6d.shape)
kde_6d = KDE(data_6d).fit_data()
pdf = kde_6d.get_pdf(np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]))
print(pdf)