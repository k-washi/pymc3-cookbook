import numpy as np
from scipy.odr import odrpack as odr
from scipy.odr import models
import matplotlib.pyplot as plt

# Sin 関数
np.random.seed(5)
x = np.random.uniform(0, 10, size=20)
y = np.random.normal(np.sin(x), 0.2) #正規分布, (平均, 分散)

#scipyによるフィッティング
polyFunc = models.polynomial(5) #5次
data = odr.Data(x, y)
myodr = odr.ODR(data, polyFunc, maxit=200)

myodr.set_job(fit_type=2) #最小二乗法

fit = myodr.run()

coeff = fit.beta[::-1]
err = fit.sd_beta[::-1]
print(coeff) #[-2.48756876e-05 -1.07573181e-02  2.09111045e-01 -1.21915241e+00, 2.11555200e+00 -1.08779827e-01] #x50
#[-4.02198060e-06, -1.58766345e-02,  3.01196923e-01, -1.78735664e+00, 3.45493019e+00, -1.11299224e+00] #x20
print(err)

px = np.linspace(0, 10, 1000)

plt.figure
plt.scatter(px, np.poly1d(coeff)(px), s=3, c='r')
plt.scatter(px, np.sin(px), s=3, c='g')

plt.plot(x, y, 'o')
plt.xlabel('x', fontsize=16)
plt.ylabel('sin(x)', fontsize=16, rotation=90)
plt.ylim([-1.5, 1.5])
plt.savefig('./qiita/c1/scipyFit.png')