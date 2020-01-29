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
ysize = 4000
y = np.random.choice(x, size=ysize, p=npdf)

#pymc用のstick break
def stick_breaking_DP(a, K):
  b = pm.Beta('B', 1., a, shape=K)
  w = b * pm.math.concatenate([[1.], tt.extra_ops.cumprod(1. - b)[:-1]])
  return w


K = 20

with pm.Model() as model:
  a = pm.Gamma('a', 1., 1.)
  w = pm.Deterministic('w', stick_breaking_DP(a, K))
  mu = pm.Normal('mu', mu=y.mean(), sd=10, shape=K)
  sd = pm.HalfNormal('sd', sd=10, shape=K)

  y = pm.NormalMixture('y', w=w, mu=mu, sd=sd, observed=y)

  trace = pm.sample(1000, chains=1)

#chain = trace[95000:]
chain = trace[700:][::3]

plt.figure()
pm.traceplot(trace)
plt.savefig("./qiita/c2/img005.png")
print(pm.summary(chain, var_names=['a', 'mu', 'sd']))

plt.figure()
pw = np.arange(K)
plt.plot(pw, chain['w'].mean(axis=0), 'o-')
plt.xticks(pw, pw+1)
plt.xlabel('Component')
plt.ylabel('GP weight')
plt.savefig("./qiita/c2/img006.png")

