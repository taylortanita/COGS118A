import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np
data = sio.loadmat('modified_data.mat')
x = data['x'].reshape([-1, 1]) 
y = data['y'].reshape([-1, 1]) 
X = np.hstack((np.ones((len(x),1)),np.power(x,1),np.power(x,2)))

lamb = 0.0001
converge = 0.001
w = np.array([[0.0],[0.0],[0.0]])
dif = 1

weight = 0.9
while (dif >= converge):
  l2 = np.dot(np.dot(X.transpose(),X),w) - np.dot(X.transpose(), y)
  l2 = l2*weight
 
  l1 = np.sum(np.sign(np.dot(X,w)-y)*X, axis=0)[np.newaxis].transpose()
  l1 = l1*(1-weight)

  w_old = w
  w = w - lamb*(l2+l1)
  dif = np.sum(np.absolute(w-w_old))

plt.plot(x,y)
plt.grid()
plt.hold(True)
plt.plot(x,w[0]+w[1]*x+w[2]*np.power(x,2))
plt.title('Regression: 0.9')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('a4_p2_0.9.png')
plt.clf()
print('w: (' + str(w[0]) + ', ' + str(w[1]) + ', ' + str(w[2]) + ')')

###
weight = 0.5
w = np.array([[0.0],[0.0],[0.0]])
dif = 1

while (dif >= converge):
  l2 = np.dot(np.dot(X.transpose(),X),w) - np.dot(X.transpose(), y)
  l2 = l2*weight

  l1 = np.sum(np.sign(np.dot(X,w)-y)*X, axis=0)[np.newaxis].transpose()
  l1 = l1*(1-weight)

  w_old = w 
  w = w - lamb*(l2+l1)
  dif = np.sum(np.absolute(w-w_old))

plt.plot(x,y)
plt.grid()
plt.hold(True)
plt.plot(x,w[0]+w[1]*x+w[2]*np.power(x,2))
plt.title('Regression: 0.5')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('a4_p2_0.5.png')
plt.clf()
print('w: (' + str(w[0]) + ', ' + str(w[1]) + ', ' + str(w[2]) + ')')

###
weight = 0.1
w = np.array([[0.0],[0.0],[0.0]])
dif = 1

while (dif >= converge):
  l2 = np.dot(np.dot(X.transpose(),X),w) - np.dot(X.transpose(), y)
  l2 = l2*weight

  l1 = np.sum(np.sign(np.dot(X,w)-y)*X, axis=0)[np.newaxis].transpose()
  l1 = l1*(1-weight)

  w_old = w 
  w = w - lamb*(l2+l1)
  dif = np.sum(np.absolute(w-w_old))

plt.plot(x,y)
plt.grid()
plt.hold(True)
plt.plot(x,w[0]+w[1]*x+w[2]*np.power(x,2))
plt.title('Regression: 0.1')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('a4_p2_0.1.png')
plt.clf()
print('w: (' + str(w[0]) + ', ' + str(w[1]) + ', ' + str(w[2]) + ')')
