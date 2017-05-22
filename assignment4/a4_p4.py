import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np
import math

def h(x):
  return 1/(1+np.exp(-x))

train = sio.loadmat('train.mat')
x1 = train['x1'].reshape([-1, 1]) 
x2 = train['x2'].reshape([-1, 1]) 
y = train['y'].reshape([-1, 1]) 
X = np.hstack((np.ones((len(x1),1)),x1,x2))
length = len(x1)

mu0 = np.array([[0.0,0.0]])
mu1 = np.array([[0.0,0.0]])
size0 = 0
size1 = 0
array0 = np.array([[0.0,0.0]])
array1 = np.array([[0.0,0.0]])

for i in range(0,length):
  if (y[i] == 0):
    mu0[0][0] = mu0[0][0] + x1[i]
    mu0[0][1] = mu0[0][1] + x2[i]
    size0 = size0 + 1
    array0 = np.append(array0,np.array([[x1[i][0],x2[i][0]]]),axis=0)
  else:
    mu1[0][0] = mu1[0][0] + x1[i]
    mu1[0][1] = mu1[0][1] + x2[i]
    size1 = size1 + 1
    array1 = np.append(array1,np.array([[x1[i][0],x2[i][0]]]),axis=0)

mu0[0][0] = mu0[0][0]/size0
mu0[0][1] = mu0[0][1]/size0
mu1[0][0] = mu1[0][0]/size1
mu1[0][1] = mu1[0][1]/size1

print('mu0 = (' + str(mu0[0][0]) + ', ' + str(mu0[0][1]) + ')')
print('mu1 = (' + str(mu1[0][0]) + ', ' + str(mu1[0][1]) + ')')

#part b
sigma0 = np.array([[0.0,0.0],[0.0,0.0]])
for i in range(0,size0):
  l = array0[i]-mu0
  prod = np.dot(np.transpose(l),l)
  sigma0 = sigma0 + prod
sigma0 = sigma0/size0

sigma1 = np.array([[0.0,0.0],[0.0,0.0]])
for i in range(0,size1):
  l = array1[i]-mu1
  prod = np.dot(np.transpose(l),l)
  sigma1 = sigma1 + prod
sigma1 = sigma1/size1

print('covariance matrix for sigma0 = ' + str(sigma0))
print('covariance matrix for sigma1 = ' + str(sigma1))

#part c
sw = sigma1 + sigma0
sb = mu1 - mu0
sw_i = np.linalg.inv(sw)
sb_t = sb.transpose()
w_star = np.dot(sw_i,sb_t)
w_den = np.linalg.norm(w_star)
w_star = w_star/w_den

print('w_star: (' + str(w_star[0][0]) + ', ' + str(w_star[1][0]) + ')')

#part d
X = np.hstack((x1,x2))
xx = np.dot(X,w_star)
size = X.size/2
point = np.array([[0.0,0.0]])
for i in range(0,size):
  if i == 0:
    point[0] = w_star.transpose()*xx[i]
  else:
    point = np.append(point, np.array(w_star.transpose()*xx[i]),axis=0)

plt.plot(point[:,0],point[:,1],'k')
plt.plot(array0[:,0],array0[:,1],'ro')
plt.plot(array1[:,0],array0[:,1],'bo')
plt.grid()
plt.hold(True)
plt.title('Linear Discriminative Analysis')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('a4_p4.png')
plt.clf()
