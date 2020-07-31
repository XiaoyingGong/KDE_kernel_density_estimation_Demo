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
print(data_6d.shape)
kde_6d = KDE(data_6d)
pdf = kde_6d.get_pdf(np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]))
print(pdf)