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

data_2d = generate_data.mixture_normal_2d(2000)
kde_2d = KDE(data_2d).fit_data()
kde_2d.draw_result()
plt.show()