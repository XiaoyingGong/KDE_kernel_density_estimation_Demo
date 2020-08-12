# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/31 16:47  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import kde_myself
# np.random.seed(10)
# x = np.random.normal(0, 10, 1000)
#
# min_max_scaler = preprocessing.MinMaxScaler()
# X_minMax = min_max_scaler.fit_transform(x.reshape(-1,1)).flatten()
#
#
# h = kde_myself.cal_h(X_minMax.reshape(-1, 1))
# pdf_1, _ = kde_myself.get_pdf(1000**(-1/5), x, x)
# pdf_2, _ = kde_myself.get_pdf(1000**(-1/5), X_minMax, X_minMax)
# pdf_3, _ = kde_myself.get_pdf_multivariate(h, X_minMax.reshape(-1, 1), X_minMax.reshape(-1, 1))
#
#
# min_max_scaler_idx = np.argsort(x)
#
# x = x[min_max_scaler_idx]
# X_minMax = X_minMax[min_max_scaler_idx]
#
# pdf_1 = pdf_1[min_max_scaler_idx]
# pdf_2 = pdf_2[min_max_scaler_idx]
# pdf_3 = pdf_3[min_max_scaler_idx]
#
# plt.figure("figure_1")
# plt.hist(x, 20, density=True)
# plt.plot(x, pdf_1)
#
# plt.figure("figure_2")
# plt.plot(X_minMax, pdf_2)
# plt.hist(X_minMax, 20, density=True)
#
# plt.figure("figure_3")
# plt.plot(X_minMax, pdf_3)
# plt.hist(X_minMax, 20, density=True)
# plt.show()

np.random.seed(10)
x_1 = np.random.multivariate_normal([4, 4], [[1, 0], [0, 1]], 1000)
x_2 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
x = np.vstack((x_1, x_2))

min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(x)

X, Y = np.mgrid[0:1:0.01, 0:1:0.01]
X_flatten = X.flatten().reshape(-1, 1)
Y_flatten = Y.flatten().reshape(-1, 1)
data_plot = np.hstack((X_flatten, Y_flatten))
print(X.shape)

h = kde_myself.cal_h(X_minMax.reshape(-1, 1))
Z, _ = kde_myself.get_pdf_multivariate(h, X_minMax, data_plot)

Z = Z.reshape(100, 100)

plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[0, 1, 0, 1])

plt.scatter(X_minMax[:, 0], X_minMax[:, 1], c='black', s=3)
plt.axis("off")
plt.show()