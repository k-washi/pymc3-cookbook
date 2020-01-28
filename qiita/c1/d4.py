import pymc3 as pm
import numpy as np
import theano.tensor as tt
import scipy.stats as stats
import matplotlib.pyplot as plt

# Sin 関数
np.random.seed(5)
xmin = 0
xmax = 10
xsize = 20
x = np.random.uniform(xmin, xmax, size=xsize)
y = np.random.normal(np.sin(x), 0.2) #正規分布, (平均, 分散)

#GP 事前分布: f(x) ~ GP(u=(u1, u2, ...), k(x, x')), u1, u2 .. = 0と仮定
#尤度: p(y|x, f(x)) ~ N(f, σ^2I)
#GP 事後分布 p(f(x)|x,y) ~GP(U,∑)

def squareDistance(x, y):
  return np.array([[(x[i]-y[j])**2 for i in range(len(x))] for j in range(len(y))])

points = np.linspace(xmin, xmax, xsize) #参照点

with pm.Model() as model:
  #事前分布
  mu = np.zeros(xsize)
  eta = pm.HalfCauchy('eta', 5)
  rho = pm.HalfCauchy('rho', 5)
  sigma = pm.HalfCauchy('sigma', 5)

  D = squareDistance(x, x)
  K = tt.fill_diagonal(eta * pm.math.exp(-rho * D), eta + sigma) #i=jの位置にeta+sigmaを格納

  gpPri = pm.MvNormal('obs', mu, cov=K, observed=y)

  #事後分布の計算(ガウス分布の事後分布は、式で計算可能)
  

  K_ss = eta * pm.math.exp(-rho * squareDistance(points, points))
  K_s = eta * pm.math.exp(-rho * squareDistance(x, points))

  MuPost = pm.Deterministic('muPost', pm.math.dot(pm.math.dot(K_s, tt.nlinalg.matrix_inverse(K)), y)) #K_s.T inv(K) y
  SigmaPost = pm.Deterministic('sigmaPost', K_ss - pm.math.dot(pm.math.dot(K_s, tt.nlinalg.matrix_inverse(K)), K_s.T))

  step = pm.Metropolis()
  trace = pm.sample(10000,step=step) 
  #trace = pm.sample(5000) 

chain = trace[5000:]
pmu = chain['muPost'][::5]
psig = chain['sigmaPost'][::5]
#chain = trace[3000:]

plt.figure()
pm.traceplot(trace, ['eta', 'rho', 'sigma'])
plt.savefig("./qiita/c1/img002.png")
#pm.summary(chain)

plt.figure()
py = pmu.mean(axis=0)
plt.plot(points, py, 'r-', linewidth=2)

for i in range(100):
  idx = np.random.randint(0, pmu.shape[0])
  py = np.random.multivariate_normal(pmu[idx], psig[idx])
  plt.plot(points, py, 'r-', alpha=0.1, linewidth=1)
print(points)
px = np.linspace(xmin, xmax, 1000)
#preM = [-2.48756876e-05, -1.07573181e-02,  2.09111045e-01, -1.21915241e+00, 2.11555200e+00, -1.08779827e-01] #x50
preM = [-4.02198060e-06, -1.58766345e-02,  3.01196923e-01, -1.78735664e+00, 3.45493019e+00, -1.11299224e+00] #x20
preM = preM[::-1]
py = preM[0] + preM[1] * px + preM[2]* px**2 + preM[3] * px**3 + preM[4] * px**4 + preM[5] *px**5
plt.plot(px, py, 'b-', linewidth=1)

plt.plot(px, np.sin(px), 'g-', linewidth=1)
plt.plot(x, y, 'o')
plt.xlabel('x', fontsize=16)
plt.ylabel('sin(x)', fontsize=16, rotation=90)
#plt.ylim([-1.5, 1.5])
plt.savefig('./qiita/c1/img003.png')
