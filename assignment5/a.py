from a5 import *
def test_gamma(x,y,x_test,y_test,c):
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
  error1 = 0
  error2 = 0
  clf = svm.SVC(C=c)
  clf1 = svm.SVC(gamma=.1,C=c)
  clf2 = svm.SVC(gamma=.01,C=c)
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

    clf1.fit(xx,yy)
    predict1 = clf1.predict(to_testx)
    e1 = test(predict1,to_testy)
    error1 = error1 + e1 

    clf2.fit(xx,yy)
    predict2 = clf2.predict(to_testx)
    e2 = test(predict2,to_testy)
    error2 = error2 + e2 

  error = error/5
  error1 = error1/5
  error2 = error2/5
 
  print('error: ' + str(error)) 
  print('error1: ' + str(error1)) 
  print('error2: ' + str(error2)) 

  g = 0.0
  if (error1 < error):
    predict_test = clf1.predict(x_test)
    e = test(predict_test,y_test)
    return ('.1', e)
  if (error2 < error):
    predict_test = clf2.predict(x_test)
    e = test(predict_test,y_test)
    return ('.01', e)
  else:
    predict_test = clf.predict(x_test)
    e = test(predict_test,y_test)
    return('default', e)
