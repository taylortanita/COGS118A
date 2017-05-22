from a5 import *

# setting up the data for the problems
# PROBLEM 1
problem = 1
data = sio.loadmat('ionosphere.mat')
x = data['X']
y = data['Y']
y = convert_array(y,problem)

# PROBLEM 2
problem = 2 
data2 = sio.loadmat('fisheriris.mat')
meas = data2['meas']
species = data2['species']
species = convert_array(species,problem)


# PROBLEM 3
problem = 3 
data3 = sio.loadmat('arrhythmia.mat')
X = data3['X']
X = convert_x(X)
Y = data3['Y']
Y = convert_array(Y,problem)

# method to perform svm on data that has been altered to correct form
def output(x,y):
# using different percentages, error for linear svm and rbf kernal are reported
  percent = 0.8

# need to run experiment multiple times- I ran it 10 times each and averaged the
# data.
# for gamma, I took the most frequently reported gamma opposed tot he average
  (x_train,y_train,x_test,y_test) = select_data(x,y,percent)
  C8 = 0.0; error8 = 0.0; C8_rbf = 0.0; error8_rbf = 0.0; g8 = ''

# to figure out most frequently reported data, list set up 
  gam = [('default',0),('.1',0),('.01',0)]

# perform experiment 10 times
  for i in range(10):
#   pick C for Linear and RBF kernal
    (iC8,ierror8) = pick_C(x_train,y_train,x_test,y_test,1)
    (iC8_rbf,ierror8_rbf) = pick_C(x_train,y_train,x_test,y_test,0)
    (ig8, ierror8_rbf) = test_gamma(x_train,y_train,x_test,y_test,iC8_rbf)

#   updating gamma list
    C8 = C8 + iC8
    error8 = error8 + ierror8
    C8_rbf = C8_rbf + iC8_rbf
    error8_rbf = error8_rbf + ierror8_rbf
    if (ig8 == 'default'):
      (_,v) = gam[0]
      gam[0] = ('default',(v+1))
    if (ig8 == '.1'):
      (_,v) = gam[1]
      gam[1] = ('.1',(v+1))
    if (ig8 == '.01'):
      (_,v) = gam[2]
      gam[2] = ('.01',(v+1))

# averaging data out
  C8 = C8/10.0
  error8 = error8/10.0
  C8_rbf = C8_rbf/10.0
  error8_rbf = error8_rbf/10.0
  (g8,_) = max(gam, key=lambda x:x[1])
  
  print('\n')
  print('Linear SVM at 80/20: C = ' + str(C8) + ', error = ' + str(error8))
  print('RBF SVM at 80/20: C = '+str(C8_rbf)+', gamma = '+g8+', error = '+str(error8_rbf))
  print('\n')

###
# same exact code used for .6 an .4 as .8
###
  percent = 0.6
  (x_train,y_train,x_test,y_test) = select_data(x,y,percent)
  C6 = 0.0; error6 = 0.0; C6_rbf = 0.0; error6_rbf = 0.0; g6 = ''
  gam = [('default',0),('.1',0),('.01',0)]
  for i in range(10):
    (iC6,ierror6) = pick_C(x_train,y_train,x_test,y_test,1)
    (iC6_rbf,ierror6_rbf) = pick_C(x_train,y_train,x_test,y_test,0)
    (ig6, ierror6_rbf) = test_gamma(x_train,y_train,x_test,y_test,iC6_rbf)

    C6 = C6 + iC6
    error6 = error6 + ierror6
    C6_rbf = C6_rbf + iC6_rbf
    error6_rbf = error6_rbf + ierror6_rbf
    if (ig6 == 'default'):
      (_,v) = gam[0]
      gam[0] = ('default',(v+1))
    if (ig6 == '.1'):
      (_,v) = gam[1]
      gam[1] = ('.1',(v+1))
    if (ig6 == '.01'):
      (_,v) = gam[2]
      gam[2] = ('.01',(v+1))
  C6 = C6/10.0
  error6 = error6/10.0
  C6_rbf = C6_rbf/10.0
  error6_rbf = error6_rbf/10.0
  (g6,_) = max(gam, key=lambda x:x[1])
  (C6,error6) = pick_C(x_train,y_train,x_test,y_test,1)
  (C6_rbf,error6_rbf) = pick_C(x_train,y_train,x_test,y_test,0)
  (g6, error6_rbf) = test_gamma(x_train,y_train,x_test,y_test,C6_rbf)
  print('Linear SVM at 60/40: C = ' + str(C6) + ', error = ' + str(error6))
  print('RBF SVM at 60/40: C = '+str(C6_rbf)+', gamma = '+g6+', error = '+str(error6_rbf))
  print('\n')

  percent = 0.4
  (x_train,y_train,x_test,y_test) = select_data(x,y,percent)
  C4 = 0.0; error4 = 0.0; C4_rbf = 0.0; error4_rbf = 0.0; g4 = ''
  gam = [('default',0),('.1',0),('.01',0)]
  for i in range(10):
    (iC4,ierror4) = pick_C(x_train,y_train,x_test,y_test,1)
    (iC4_rbf,ierror4_rbf) = pick_C(x_train,y_train,x_test,y_test,0)
    (ig4, ierror4_rbf) = test_gamma(x_train,y_train,x_test,y_test,iC4_rbf)

    C4 = C4 + iC4
    error4 = error4 + ierror4
    C4_rbf = C4_rbf + iC4_rbf
    error4_rbf = error4_rbf + ierror4_rbf
    if (ig4 == 'default'):
      (_,v) = gam[0]
      gam[0] = ('default',(v+1))
    if (ig4 == '.1'):
      (_,v) = gam[1]
      gam[1] = ('.1',(v+1))
    if (ig4 == '.01'):
      (_,v) = gam[2]
      gam[2] = ('.01',(v+1))
  C4 = C4/10.0
  error4 = error4/10.0
  C4_rbf = C4_rbf/10.0
  error4_rbf = error4_rbf/10.0
  (g4,_) = max(gam, key=lambda x:x[1])
  (C4,error4) = pick_C(x_train,y_train,x_test,y_test,1)
  (C4_rbf,error4_rbf) = pick_C(x_train,y_train,x_test,y_test,0)
  (g4, error4_rbf) = test_gamma(x_train,y_train,x_test,y_test,C4_rbf)
  print('Linear SVM at 40/60: C = ' + str(C4) + ', error = ' + str(error4))
  print('RBF SVM at 40/60: C = '+str(C4_rbf)+', gamma = '+g4+', error = '+str(error4_rbf))
  print('\n')


# calling output on the data
print('Problem 1\n')
output(x,y)
print('Problem 2\n')
output(meas,species)
print('Problem 3\n')
output(X,Y)
