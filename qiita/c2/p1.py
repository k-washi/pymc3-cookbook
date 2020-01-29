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

"""
plt.figure()
plt.plot(x, npdf, 'g-', linewidth=2)
plt.hist(y, density=True, bins=50, alpha=0.3)


plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16, rotation=0)
#plt.ylim([-0.1, npdf.max()+0.1])
plt.savefig('./qiita/c2/img001.png')

"""

cluster = 3
'''
#隠れ変数zがあるため計算が遅い。∫p(y|z,θ)dz -> p(y|θ) 
with pm.Model() as model:
  p = pm.Dirichlet('p', a=np.ones(cluster))
  z = pm.Categorical('z', p=p, shape=y.shape[0])

  mu = pm.Normal('mu', mu=y.mean(), sd=10, shape=cluster)
  sd = pm.HalfNormal('sd', sd=10, shape=cluster)

  y = pm.Normal('y', mu=mu[z], sd=sd[z], observed=y)

  #step = pm.Metropolis()
  #trace = pm.sample(5000,step=step, tune=100,chains=3, random_seed=3)
  trace = pm.sample(1000)
'''
with pm.Model() as model:
  p = pm.Dirichlet('p', a=np.ones(cluster))
  mu = pm.Normal('mu', mu=y.mean(), sd=10, shape=cluster)
  sd = pm.HalfNormal('sd', sd=10, shape=cluster)

  y = pm.NormalMixture('y', w=p, mu=mu, sd=sd, observed=y)

  #step = pm.Metropolis()
  #trace = pm.sample(5000,step=step, tune=100,chains=3, random_seed=3)
  trace = pm.sample(3000, chains=1)
#chain = trace[95000:]
chain = trace[800:][::3]

plt.figure()
pm.traceplot(trace)
plt.savefig("./qiita/c2/img002.png")
print(pm.summary(chain))

"""
結果がおかしい label-switching problem
https://stan-ja.github.io/gh-pages-html/#%E6%B7%B7%E5%90%88%E5%88%86%E5%B8%83%E3%83%A2%E3%83%87%E3%83%AB%E3%81%A7%E3%81%AE%E3%83%A9%E3%83%99%E3%83%AB%E3%82%B9%E3%82%A4%E3%83%83%E3%83%81%E3%83%B3%E3%82%B0
混合分布の成分を入れ替えることができる場合、混合成分が入れ替わる可能性がある

mu[0] -4.015  4.005  -8.103    0.121      2.774    2.330       2.0     2.0       3.0     191.0   1.83
mu[1] -1.995  6.021  -8.146    4.151      4.171    3.503       2.0     2.0       3.0      35.0   1.82
mu[2]  1.992  1.992  -0.132    4.120      1.379    1.158       2.0     2.0       3.0     110.0   1.83
sd[0]  1.606  0.198   1.342    1.884      0.133    0.111       2.0     2.0       3.0     118.0   1.83
sd[1]  1.567  0.252   1.248    1.914      0.171    0.142       2.0     2.0       3.0     207.0   1.83
sd[2]  1.387  0.072   1.254    1.515      0.031    0.023       5.0     5.0       5.0     140.0   1.32

chains=1として対処(chainsを複数にする場合は、何らかの基準を設けて並び替える必要がある)
例えば、音源分離でラベリングを行う問題を扱う研究では、音源方向を基準として与えたりしている。

        mean     sd  hpd_3%  hpd_97%  mcse_mean  mcse_sd  ess_mean  ess_sd  ess_bulk  ess_tail  r_hat
mu[0] -8.019  0.070  -8.150   -7.883      0.003    0.002     703.0   703.0     703.0     475.0    NaN
mu[1]  4.009  0.098   3.828    4.179      0.005    0.003     429.0   429.0     445.0     327.0    NaN
mu[2] -0.000  0.072  -0.125    0.139      0.003    0.002     469.0   469.0     472.0     568.0    NaN
sd[0]  1.803  0.055   1.693    1.902      0.002    0.001     703.0   703.0     707.0     683.0    NaN
sd[1]  1.323  0.059   1.224    1.445      0.003    0.002     536.0   535.0     536.0     523.0    NaN
sd[2]  1.427  0.050   1.321    1.517      0.002    0.002     516.0   516.0     521.0     501.0    NaN
"""