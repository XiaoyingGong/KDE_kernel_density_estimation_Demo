# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/24 20:27  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
# -*- coding: UTF-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
import kde_1d
import generate_data

data = generate_data.standard_normal_1d(2000)
x = np.linspace(-10, 10, 1000).reshape(-1, 1)

# ----------------------------------- fit data 进行拟合 -----------------------------------
time_1 = time.time()
h = kde_1d.h_determination(data)
print("h:", h)
y = kde_1d.get_kde(x, data, h, kde_1d.gauss_kernel)
time_2 = time.time()
print("耗时为：", time_2 - time_1)

# ----------------------------------- fit data 进行绘制 -----------------------------------
plt.hist(data, bins=40, density=True)
plt.plot(x, y, c='r')
plt.show()




