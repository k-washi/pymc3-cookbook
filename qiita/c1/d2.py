import pymc3 as pm
import numpy as np
import theano.tensor as tt
import scipy.stats as stats
import matplotlib.pyplot as plt

# Sin 関数
np.random.seed(10)
x = np.random.uniform(0, 10, size=50)
y = np.random.normal(np.sin(x), 0.2) #正規分布, (平均, 分散)

#カーネルを用いた多項式回帰
#xの代わりにガウスカーネルを用いる
def kernel(x, num_knots):
  knots = np.linspace(x.min(), x.max(), num_knots)
  w = 2 #偏差σ=1
  return np.array([np.exp(-(x - k)**2/w) for k in knots])


# y = b0 + b1x + ∑rK (n=5)
kernelKnotsNum = 5

with pm.Model() as model:
  b0 = pm.Normal('b0', mu=0, sd=5)
  b1 = pm.Normal('b1', mu=0, sd=1)
  gamma = pm.Cauchy('gamma', alpha=0, beta=1, shape=kernelKnotsNum)

  mu = pm.math.dot(gamma, kernel(x, kernelKnotsNum)) + b0 + b1*x
  #sd = pm.HalfCauchy('sd', 5)
  sd = pm.Uniform('sd', 0, 10)  

  y_prod = pm.Normal('y', mu=mu, sd=sd, observed=y)

  step = pm.Metropolis()
  #trace = pm.sample(100000,step=step) 
  trace = pm.sample(5000) 

#chain = trace[95000:]
chain = trace[3000:]

plt.figure()
pm.traceplot(trace)
plt.savefig("./qiita/c1/img002.png")

plt.figure()
px = np.linspace(0, 10, 100)
k = kernel(px, kernelKnotsNum)
#print(k.shape)#(5, 100)
#print(chain['gamma'].shape)#(1000, 5)
py = np.dot(chain['gamma'].mean(axis=0), k) + chain['b0'].mean() + chain['b1'].mean() * px
plt.plot(px, py, 'r-', linewidth=2)

for i in range(50):
  idx = np.random.randint(0, len(chain['b0']))
  py = np.dot(chain['gamma'][idx], k) + chain['b0'][idx]+ chain['b1'][idx] * px 
  plt.plot(px, py, 'r-', alpha=0.1, linewidth=1)

preM = [-2.48756876e-05, -1.07573181e-02,  2.09111045e-01, -1.21915241e+00, 2.11555200e+00, -1.08779827e-01]
preM = preM[::-1]
py = preM[0] + preM[1] * px + preM[2]* px**2 + preM[3] * px**3 + preM[4] * px**4 + preM[5] *px**5
plt.plot(px, py, 'b-', linewidth=2)

plt.plot(px, np.sin(px), 'g-', linewidth=2)
plt.plot(x, y, 'o')
plt.xlabel('x', fontsize=16)
plt.ylabel('sin(x)', fontsize=16, rotation=90)
plt.ylim([-1.5, 1.5])
plt.savefig('./qiita/c1/img003.png')