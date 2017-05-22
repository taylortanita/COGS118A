import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import math

train = sio.loadmat('train.mat')
x1 = train['x1'].reshape([-1, 1])
x2 = train['x2'].reshape([-1, 1])
x0 = np.ones((len(x1),1))
y = train['y'].reshape([-1, 1])
X = np.hstack((np.ones((len(x1),1)),x1,x2))

lamb = 0.001
converge = 30
w = np.array([[0.0],[0.0],[0.0]])

def h(x):
  return 1/(1+np.exp(-x))

def classify(x):
  if x >= 0.5:
    return 1.0
  else:
    return 0.0

for i in range(0,100):
  prod = np.dot(X,w)
  h_res = np.apply_along_axis(h,0,prod)
  h_res = h_res - y
  x0_res = np.multiply(h_res,x0)
  x1_res = np.multiply(h_res,x1)
  x2_res = np.multiply(h_res,x2)
  x0_sum = np.sum(x0_res,axis=0)
  x1_sum = np.sum(x1_res,axis=0)
  x2_sum = np.sum(x2_res,axis=0)
  w_old = w
  w[0] = w_old[0] - lamb*x0_sum
  w[1] = w_old[1] - lamb*x1_sum
  w[2] = w_old[2] - lamb*x2_sum

results = np.dot(X,w)
results = np.apply_along_axis(h,0,results)

print('optimal w: ('+str(w[0][0])+', '+str(w[1][0])+', '+str(w[2][0])+')')

#testing
test = sio.loadmat('test.mat')
x1_test = test['x1'].reshape([-1, 1])
x2_test = test['x2'].reshape([-1, 1])
x0_test = np.ones((len(x1),1))
y_test = test['y'].reshape([-1, 1])
X_test = np.hstack((np.ones((len(x1_test),1)),x1_test,x2_test))
results_test = np.dot(X_test,w)
results_test = np.apply_along_axis(h,0,results_test)

s = len(x1_test)
correct = 0.0
total = 0.0
for i in range(0,s):
  if ((results_test[i]>=0.5)&(y_test[i]==1)) | ((results_test[i]<0.5)&(y_test[i]==0)):
    correct = correct + 1.0
  total = total + 1.0

incorrect = total - correct
percent_c = correct/total
percent_i = incorrect/total

print('percent correct: ' + str(percent_c*100) + '%')
print('testing error: ' + str(percent_i*100) + '%')
