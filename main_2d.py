# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/27 16:42  
# IDE：PyCharm 
# des: Have some problems in this implementation.
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
import kde_2d
import generate_data

data = generate_data.mixture_normal_2d(2000)

x_min = np.min(data[:, 0])
x_max = np.max(data[:, 0])
y_min = np.min(data[:, 1])
y_max = np.max(data[:, 1])

x, y = np.mgrid[x_min:x_max:20j, y_min:y_max:20j]
x = x.flatten().reshape(-1, 1)
y = y.flatten().reshape(-1, 1)
x = np.hstack((x, y))

h = kde_2d.h_determination(data)
h = np.array([0.01, 0.01])
print(h)
result = kde_2d.get_kde_2d(x, data, h, kde_2d.gauss_kernel)
result = result.reshape([20, 20])
plt.hist2d(x=data[:, 0], y=data[:, 1], bins=40)
x, y = np.mgrid[x_min:x_max:20j, y_min:y_max:20j]
plt.contour(x, y, result, cmap='gray_r')
plt.show()