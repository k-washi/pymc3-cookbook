import pymc3 as pm
import numpy as np
import theano.tensor as tt
import scipy.stats as stats
from scipy import optimize
import matplotlib.pyplot as plt

# Sin 関数
np.random.seed(10)
x = np.random.uniform(0, 10, size=50)
y = np.random.normal(np.sin(x), 0.2) #正規分布, (平均, 分散)

"""
plt.figure()
plt.plot(x, y, 'o')
plt.xlabel('x', fontsize=16)
plt.ylabel('sin(x)', fontsize=16, rotation=90)
plt.savefig('./qiita/c1/img001.png')
"""

#多項式回帰
#Mu = b0 * x^0 + b1 * x^1 + b2 * x^2 + b3 * x^3 という回帰モデルを構築する。
preM = [0]*6
with pm.Model() as model:
  b0 = pm.Normal('b0', mu=preM[0], sd=10)
  b1 = pm.Normal('b1', mu=preM[1], sd=5)
  b2 = pm.Normal('b2', mu=preM[2], sd=5)
  b3 = pm.Normal('b3', mu=preM[3], sd=5)
  b4 = pm.Normal('b4', mu=preM[4], sd=5)
  b5 = pm.Normal('b5', mu=preM[5], sd=5)



  #コーシー分布 x ∈ [0, ∞) https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.HalfCauchy
  sd = pm.HalfCauchy('sd', 5)   
  mu = b0 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4 + b5 * x**5
  nu = pm.Exponential('nu', 1/30)
  #mu = pm.Deterministic('mu', )

  y_pred = pm.StudentT('y_pred', mu=mu, sd=sd, nu=nu, observed=y)
  #start = pm.find_MAP(fmin=optimize.fmin_powell)
  step = pm.Metropolis()
  #trace = pm.sample(100000,step=step)
  trace = pm.sample(5000)


#chain = trace[95000:]
chain = trace[3000:]

plt.figure()
pm.traceplot(trace)
plt.savefig("./qiita/c1/img002.png")
pm.summary(chain)


px = np.linspace(0, 10, 1000)
plt.figure()
py = chain['b0'].mean() + chain['b1'].mean() * px + chain['b2'].mean() * px**2 + chain['b3'].mean() * px**3 + chain['b4'].mean() * px**4 + chain['b5'].mean() *px**5
plt.plot(px, py, 'r-', linewidth=2)

for i in range(100):
  idx = np.random.randint(0, len(chain['b0']))
  py = chain['b0'][idx] + chain['b1'][idx] * px + chain['b2'][idx]* px**2 + chain['b3'][idx] * px**3 + chain['b4'][idx] * px**4 + chain['b5'][idx] *px**5
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





