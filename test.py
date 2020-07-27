# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/27 16:12  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import time
def get_kde(x,data_array,bandwidth=0.1):
    def gauss(x):
        import math
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2))
    N=len(data_array)
    res=0
    if len(data_array)==0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x-data_array[i])/bandwidth)
    res /= (N*bandwidth)
    return res
import numpy as np
input_array=np.random.randn(20000).tolist()
bandwidth=1.05*np.std(input_array)*(len(input_array)**(-1/5))
x_array=np.linspace(min(input_array),max(input_array),50)
y_array=[get_kde(x_array[i],input_array,bandwidth) for i in range(x_array.shape[0])]

import matplotlib.pyplot as plt
plt.figure(1)
plt.hist(input_array,bins=40,density=True)
plt.plot(x_array.tolist(),y_array,color='red',linestyle='-')
plt.show()