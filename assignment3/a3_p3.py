import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(10)
plt.hist(x)
plt.savefig('x_hist.png')
plt.clf()

y = np.random.randn(10)
plt.scatter(x,y)
plt.savefig('scatter.png')
plt.clf()

xp = np.array([])
yp = np.array([])
for i in range(0,10):
  if i == 0:
    xp = np.array([3*x[i]+y[i]])
    yp = np.array([x[i]-2*y[i]])
  else:
    xp = np.append(xp, [3*x[i]+y[i]], axis=0)
    yp = np.append(yp, [x[i]-2*y[i]], axis=0)
 
plt.hist(xp)
plt.savefig('xp_hist.png')
plt.clf()

plt.hist(yp) 
plt.savefig('yp_hist.png')
plt.clf()
