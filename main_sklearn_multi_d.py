# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/31 17:12  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
from kde_sklearn_multi_d import KDE
import generate_data
import numpy as np

data_6d = generate_data.mixture_normal_6d(500)
kde_6d = KDE(data_6d)
kde_6d = kde_6d.fit_data()
pdf = kde_6d.get_pdf(np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]))
a, b = kde_6d.get_rank_pdf(0.5)
print(a)