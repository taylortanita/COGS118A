import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('data.mat')
x = data['x'].reshape([-1,1])
y = data['y'].reshape([-1,1])
X = np.hstack((np.ones((len(x),1)),np.power(x,1)))

size = X.size/2
dif = 1
lamb = 0.0001
lim = 0.001
w = np.array([[0.0],[0.0]])
while (dif >= lim):
  deriv = np.array([[0.0],[0.0]])
  for i in range(0,size):
    deriv = deriv + np.sign(np.dot(X[i][np.newaxis],w)-y[i])*X[i][np.newaxis].transpose() 
  w_old = w
  w = w - lamb*deriv
  dif = np.linalg.norm((w-w_old),1)

plt.plot(x,y)
plt.grid()
plt.hold(True)
plt.plot(x,w[0]+w[1]*x)
plt.title('Least square line fitting: Gradient Descent L1')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('prob6.png')
plt.clf()
print('Gradient Descent L1: w0: ' + str(w[0]) + ', w1: ' + str(w[1]))
