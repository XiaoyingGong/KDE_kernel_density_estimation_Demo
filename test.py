# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/28 10:45  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2

def mixture_normal(num):
    mus_1 = np.array([0, 0])
    sigmas_1 = np.array([[1, 0], [0, 1]])
    mus_2 = np.array([4, 4])
    sigmas_2 = np.array([[1, 0], [0, 1]])


    data_1 = np.random.multivariate_normal(mus_1, sigmas_1, num)
    data_2 = np.random.multivariate_normal(mus_2, sigmas_2, num)
    data = np.vstack((data_1, data_2))
    return data.T[0, :], data.T[1, :]

def uniform(num):
    np.random.seed(1)
    data = np.random.uniform(0, 1, [num, 2])
    return data.T[0, :], data.T[1, :]

m1, m2 = mixture_normal(200)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(m1, m2, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()

kernel_values = kernel(values)
print(kernel_values)