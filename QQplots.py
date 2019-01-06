import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(1234)

sampleTemp = np.random.normal(loc=0, scale=1, size=1000)
sample = np.unique(sampleTemp)
quantilesSample = np.arange(0, sample.shape[0]) / (sample.shape[0] - 1)
# get quantiles of any distribution we are interested in
quantileArea = 1. / (sample.shape[0] + 1)
quantiles = np.cumsum(np.ones(sample.shape[0]) * quantileArea)
quantiles = norm.ppf(quantiles, loc=0, scale=1)
# compare distribution quantiles with data
# sum of squared errors of the model
sse = np.sum(np.square(np.sort(sample) - quantiles))
# total of squared errors of the model
sst = np.sum(np.square(np.sort(sample) - np.mean(np.sort(sample))))
# coefficient of determination
r2 = 1. - (sse / sst)
# plot
figQQ, axQQ = plt.subplots()
axQQ.scatter(x=quantiles, y=np.sort(sample), label='Sample')
axQQ.plot(quantiles, quantiles, label='Theoretically expected', color='red')
axQQ.set_xlabel('Theoretical Quantiles')
axQQ.set_ylabel('Sample Quantiles')
axQQ.legend()
axQQ.text(0.8, 0.1, '$R^2 = {:.3f}$'.format(r2), ha='center', va='center',
          fontdict={'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal'},
          transform=axQQ.transAxes)