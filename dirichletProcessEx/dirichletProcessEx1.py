import pymc3
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt


RAND_SEED = 5123455
np.random.seed(RAND_SEED)

N = 20
K = 30

#alpha -> ∞　ディリクレ過程が基底分布に収束
alpha = 50
Prob0 = sp.stats.norm #基底分布 N(0,1)
omega = Prob0.rvs(size=(N, K)) 

#stick-breaking process

#rvs (Random variates)確率変数をnp.array((N,K))で作成
beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
w = np.empty_like(beta)

w[:, 0] = beta[:,0]
w[:, 1:] = beta[:, 1:] * (1 -beta[:, :-1]).cumprod(axis=1)
print(w[0])

"""
# a<b
>>> a = np.array([1,2,3,4,5])
>>> b = np.array([1,2])
>>> np.less.outer(a,b)
array([[False,  True],
       [False, False],
       [False, False],
       [False, False],
       [False, False]])

"""

x_plot = np.linspace(-3, 3, 200)
#(N, K, 200)
diracDeltaMeasure = np.less.outer(omega, x_plot)

#(N, 200)
#累積分布
smaple_cdfs = (w[..., np.newaxis] * diracDeltaMeasure).sum(axis = 1)

plt.figure()
plt.plot(x_plot, smaple_cdfs[0], c="gray", alpha=0.75)
plt.plot(x_plot, smaple_cdfs[1:].T, c="gray", alpha=0.75)
plt.plot(x_plot, Prob0.cdf(x_plot))

plt.show()