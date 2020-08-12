# author: 龚潇颖(Xiaoying Gong)
# date： 2020/8/11 14:43  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt

def cal_h(data):
    N, d = data.shape
    return N ** (- 1 / (d + 4))

def gaussian_kernel(x):
    kernel_val = (1 / np.sqrt(2 * np.pi)) * np.exp(- x**2 / 2)
    return kernel_val


def get_pdf(h, data, x):
    data_len = len(data)
    x_len = len(x)
    dist = np.sqrt((np.tile(x.reshape(-1, 1), (1, data_len)) - np.tile(data, (x_len, 1))) ** 2)
    pdf = (1/(data_len * h)) * np.sum(gaussian_kernel(dist/h), axis=1)
    print(pdf)
    number = np.sum(gaussian_kernel(dist/h), axis=1)
    return pdf, number

def get_pdf_multivariate(h, data, x):
    data_len, d = data.shape
    x_len = len(x)
    x_reshape = x.reshape((-1, 1, d))
    data_reshape = data.reshape((1, -1, d))
    x_reshape = np.tile(x_reshape, (1, data_len, 1))
    data_reshape = np.tile(data_reshape, (x_len, 1, 1))
    dist = (x_reshape - data_reshape) / h
    number = np.sum(np.prod(gaussian_kernel(dist), axis=2), axis=1)
    pdf = (1/(data_len * h**d)) * number
    return pdf, number

# a = np.random.normal(0, 1, 1000)
# h = 1000**(-1/5)
# print(h)
# b, number = get_pdf(h, a, a)
#
# sorted_a = np.argsort(a)
# a = a[sorted_a]
# b = b[sorted_a]
#
# print(number)
#
# plt.plot(a, b)
# plt.hist(a, 40, density=True)
# plt.show()
if __name__ == '__main__':
    a = np.array([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    print(np.prod(a, axis=2))