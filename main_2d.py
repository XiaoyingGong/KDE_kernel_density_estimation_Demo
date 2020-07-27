# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/27 16:42  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt

mus = np.array([2, 2])
sigmas = np.array([[2, 0], [0, 2]])
data = np.random.multivariate_normal(mus, sigmas, 100000)
print(data)

plt.hist2d(x=data[:, 0], y=data[:, 1], bins=50)
plt.show()