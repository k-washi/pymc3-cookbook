import pymc3 as pm
import numpy as np
import theano.tensor as tt
import scipy.stats as stats
from scipy import optimize
import matplotlib.pyplot as plt

np.random.seed(53536)

xmin = -15.
xmax = 10.
xsize = 200
x = np.linspace(xmin, xmax, xsize)
pi_k = np.array([0.2, 0.5, 0.3])
loc_x = np.array([-8, 0, 4])

norm1 = stats.norm.pdf(x, loc=loc_x[0], scale=1.8) * pi_k[0]
norm2 = stats.norm.pdf(x, loc=loc_x[1], scale=1.5) * pi_k[1]
norm3 = stats.norm.pdf(x, loc=loc_x[2], scale=1.3) * pi_k[2]

npdf = norm1 + norm2 + norm3
npdf /= npdf.sum()

#確率分布の確率に則って値(x)を取得
y = np.random.choice(x, size=4000, p=npdf)

#棒折過程: stick breaking prcess
def stick_breaking(a, h, k):
  '''
  a:集中度
  h:基底分布(scipy dist)
  K: コンポーネント数

  Return
  locs : 位置(array)
  w: 確率(array)
  '''
  s = stats.beta.rvs(1, a, size=K)
 #ex : [0.02760315 0.1358357  0.02517414 0.11310199 0.21462781]
  w = np.empty(K)
  w = s * np.concatenate(([1.], np.cumprod(1 - s[:-1])))
  #ex: 0.02760315 0.13208621 0.0211541  0.09264824 0.15592888]
  # if i == 1, s , elif i > 1, s∑(1-sj) (j 1 -> i-1)

  locs = H.rvs(size=K)
  return locs, w

'''
K = 200
H = stats.norm
concentrateParam = [1, 10, 100, 1000]
_, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 5))
ax = np.ravel(ax)
for idx, a in enumerate(concentrateParam):
  locs, w = stick_breaking(a, H, K)
  
  ax[idx].vlines(locs, 0, w, color='C0')
  ax[idx].set_title('a = {}'.format(a))

plt.tight_layout()
plt.savefig("./qiita/c2/img003.png")
'''

K = 5
H = stats.norm
a = 1000

x = np.linspace(-5, 5, 250)
x_ = np.array([x] * K).T #(250, K)

locs, w = stick_breaking(a, H, K)

dist = stats.norm(locs, 0.5)
plt.plot(x, np.sum(dist.pdf(x_) * w, 1), 'C0', lw = 2)
plt.plot(x, dist.pdf(x_) * w, 'k--', alpha=0.5)
plt.savefig("./qiita/c2/img004.png")
