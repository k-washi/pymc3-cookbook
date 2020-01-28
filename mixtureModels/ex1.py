import pandas as pd
from matplotlib import pyplot as plt
import pymc3 as pm
import numpy as np

cs = pd.read_csv("./data/chemical_shifts_theo_exp.csv")

cs_exp = cs['exp']

"""
plt.hist(cs_exp, density=True, bins=30, alpha=0.3)
plt.show()
"""

clusters = 2
"""
with pm.Model() as model_kg:
  p = pm.Dirichlet('p', a=np.ones(clusters))
  z = pm.Categorical('z', p=p, shape=len(cs_exp))
  means = pm.Normal('means', mu=cs_exp.mean(), sd = 10,shape=clusters)
  sd = pm.HalfNormal('sd', sd=10)

  y = pm.Normal('y', mu=means[z], sd = sd, observed=cs_exp)
  trace_kg = pm.sample()
"""
print("Model set")
with pm.Model() as model_kg:
  p = pm.Dirichlet('p', a=np.ones(clusters))
  #z = pm.Categorical('z', p=p, shape=len(cs_exp))
  means = pm.Normal('means', mu=cs_exp.mean(), sd = 10,shape=clusters)
  sd = pm.HalfNormal('sd', sd=10)

  y = pm.NormalMixture('y', w=p, mu=means, sd = sd, observed=cs_exp)
  print("Sample start")
  trace_kg = pm.sample(random_seed=123)


pm.traceplot(trace_kg)
plt.show()