import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import tree 
import copy
import math


def convert(elmt):
  if (elmt[0][0] == u'b'):
    return 1
  else:
    return 0

def convert_array(y):
  yy = np.array([[-1]])
  (length,_) = y.shape
  for i in range(length):
    yy = np.append(yy,[[convert(y[i])]],axis=0)
  return yy[1:]

# method that selects the percentage specified for training and testing purposed
# returns a tuple with the training and testing data
def select_data(x,y,percent):
  (size,width) = x.shape
  train_size = np.ceil(size*percent)

  data = np.concatenate((x,y),axis=1)
  np.random.shuffle(data)
  x = data[:,:(width)]
  y = data[:,[(width)]]

  x_train = x[:train_size]
  x_test = x[train_size:]
  y_train = y[:train_size]
  y_test = y[train_size:]

  return (x_train,y_train,x_test,y_test)

# helper method for test_c
# takes in two lists and returns the percent error 
# i.e. the percentage of time where the two lists dffer
def test(predict,y_test):
  y = y_test.ravel()
  total = len(predict)
  errors = 0.0
  for i in range(total):
    if (predict[i] != y[i]):
      errors = errors + 1.0
  return errors/total


###### METHODS FOR Q3 ######

# method to compute error using specific D
# reports the error associated with that D value
def compute_error(x,y,D):
  (size,width) = x.shape
  mod = size % 5
  xx = x[mod:]
  yy = y[mod:]

  x_part = np.split(xx,5)
  y_part = np.split(yy,5)

  x_part[0] = np.concatenate((x_part[0],x[:mod]),axis=0)
  y_part[0] = np.concatenate((y_part[0],y[:mod]),axis=0)

  # 5-fold cross validation
  error = 0
# use appropriate svm type
  clf = tree.DecisionTreeClassifier(max_depth=D)
  if (D == 400):
    clf = tree.DecisionTreeClassifier()
  for i in range(5):
    X = copy.deepcopy(x_part)
    to_testx = X.pop(i)
    xx = np.concatenate((X[0],X[1]),axis=0)
    xx = np.concatenate((xx,X[2]),axis=0)
    xx = np.concatenate((xx,X[3]),axis=0)

    Y = copy.deepcopy(y_part)
    to_testy = Y.pop(i)
    yy = np.concatenate((Y[0],Y[1]),axis=0)
    yy = np.concatenate((yy,Y[2]),axis=0)
    yy = np.concatenate((yy,Y[3]),axis=0)

    clf.fit(xx,yy)
    predict = clf.predict(to_testx)
    e = test(predict,to_testy)
    error = error + e

  error = error/5
#  print('error = '+str(error))
  return error

# runs cross validation multiple times 
def run(x_train,y_train,D):
  error = 0.0
  for i in range(30):
    error = error + compute_error(x_train,y_train,D)
  error = error/30
  return error

# method to pick which D is best
# takes in training data and list of D values to test out
# reports the D value
def pick_D(x_train,y_train,D_choices):
  error = 1
  D = D_choices[0]
  iterate = len(D_choices)
  for i in range(iterate):
    e = run(x_train,y_train,D_choices[i])
#    print('for D = '+str(D_choices[i])+', e = '+str(e))
    if (e < error):
      error = e
      D = D_choices[i]
  return (D,error)

###### METHODS FOR Q4 ######

def distance(a,b):
  d = numpy.linalg.norm(a-b)
  return d

# returns k nearest neighbors of target
def get_neighbors(x_data,y_data,target,k):
  (length,_) = x_data.shape
# distances contains tuples of (euclid norm, class)
  distances = []
  for i in range(length):
    d = np.linalg.norm(target-x_data[i])
    distances = distances + [(d,y_data[i])]
  distances.sort(key=lambda x:x[0])
  num_0 = 0
  num_1 = 0
  for i in range(k):
    (dis,clas) = distances[i]
    if (clas == 0):
      num_0 = num_0 + 1
    else:
      num_1 = num_1 + 1
  if (num_0 > num_1):
    return 0
  else:
    return 1

# returns list of predicted classes
def predict_class(x_data,y_data,x_target,k):
  (length,_) = x_target.shape
  classes = []
  for i in range(length):
    c = get_neighbors(x_data,y_data,x_target[i],k)
    classes = classes + [c]
  return classes


# cross validation
# returns error with associated k
def run_k(x,y,k):
  (size,width) = x.shape
  mod = size % 5 
  xx = x[mod:]
  yy = y[mod:]

  x_part = np.split(xx,5)
  y_part = np.split(yy,5)

  x_part[0] = np.concatenate((x_part[0],x[:mod]),axis=0)
  y_part[0] = np.concatenate((y_part[0],y[:mod]),axis=0)

  # 5-fold cross validation
  error = 0.0
  for i in range(5):
    X = copy.deepcopy(x_part)
    to_testx = X.pop(i)
    xx = np.concatenate((X[0],X[1]),axis=0)
    xx = np.concatenate((xx,X[2]),axis=0)
    xx = np.concatenate((xx,X[3]),axis=0)

    Y = copy.deepcopy(y_part)
    to_testy = Y.pop(i)
    yy = np.concatenate((Y[0],Y[1]),axis=0)
    yy = np.concatenate((yy,Y[2]),axis=0)
    yy = np.concatenate((yy,Y[3]),axis=0)

    y_predict = predict_class(xx,yy,to_testx,k)
    e = test(y_predict,to_testy)
#    print('  error: ' + str(e))
    error = error + e
  
  error = error/5
  return error

def pick_k(x_train,y_train,k_choices):
  error = 1 
  k = k_choices[0]
  iterate = len(k_choices)
  for i in range(iterate):
    e = run_k(x_train,y_train,k_choices[i])
#    print('k = '+str(k_choices[i])+', error: '+str(e))
    if (e < error):
      error = e 
      k = k_choices[i]
  return (k,error) 


# shuffling data and running many times
def multi_k(x,y,percent,k_choices):
  occ = [0,0,0,0]
  err = [0.0,0.0,0.0,0.0]

  for i in range(10):  
    (x_train,y_train,x_test,y_test) = select_data(x,y,percent)
    (k,error) = pick_k(x_train,y_train,k_choices)
  
    if(k==1):
      occ[0] = occ[0]+1
      err[0] = err[0]+error
    if(k==3):
      occ[1] = occ[1]+1
      err[1] = err[1]+error
    if(k==5):
      occ[2] = occ[2]+1
      err[2] = err[2]+error
    if(k==7):  
      occ[3] = occ[3]+1
      err[3] = err[3]+error

  print('occ: '+str(occ))
  i = occ.index(max(occ))
  final_error = err[i]/occ[i]
  k = 1
  if(i==1):
    k = 3
  if(i==2):
    k = 5
  if(i==3):
    k = 7
  return(k,final_error)
