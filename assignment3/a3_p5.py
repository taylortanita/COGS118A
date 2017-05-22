import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('data.mat')
x = data['x'].reshape([-1,1])
y = data['y'].reshape([-1,1])
X = np.hstack((np.ones((len(x),1)),np.power(x,1)))

w = np.array([[0.0],[0.0]])

## Number 5 ##
lamb = 0.0001
lim = 0.001

dif = 1
while (dif >= lim):
  a = 2*(np.dot(np.dot(X.transpose(),X),w))
  b = 2*np.dot(X.transpose(),y)
  deriv = a - b
  w_old = w
  w = w - lamb*deriv
# unsure as to how to compute dif
  dif = np.sum(np.absolute(w-w_old))

(n,m) = deriv.shape
(k,l) = w.shape
print('dimension of derivative is (' + str(n) + ',' + str(m) + ')')
print('dimension of W is (' + str(k) + ',' + str(l) + ')')

plt.plot(x,y)
plt.grid()
plt.hold(True)
plt.plot(x,w[0]+w[1]*x)
plt.title('Least square line fitting: Gradient Descent L2')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('prob5.png')
plt.clf()

(w_closed,_,_,_) = np.linalg.lstsq(X,y)
plt.plot(x,y)
plt.grid()
plt.hold(True)
plt.plot(x,w_closed[0]+w_closed[1]*x)
plt.title('Least square line fitting: Closed Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('prob5_closed.png')
plt.clf()

print('Gradient Descent: w0: ' + str(w[0]) + ', w1: ' + str(w[1]))
print('Closed Form: w0: ' + str(w[0]) + ', w1: ' + str(w[1]))

