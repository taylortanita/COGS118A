from data import *
from sklearn.neighbors import KNeighborsClassifier

# cross validation
# returns error with associated k
def run_k(x,y,k):
  (size,width) = x.shape
  mod = size % 10
  xx = x[mod:]
  yy = y[mod:]

  x_part = np.split(xx,10)
  y_part = np.split(yy,10)

  x_part[0] = np.concatenate((x_part[0],x[:mod]),axis=0)
  y_part[0] = np.concatenate((y_part[0],y[:mod]),axis=0)

  # 10-fold cross validation
  error = 0.0

  for i in range(10):
    X = copy.deepcopy(x_part)
    to_testx = X.pop(i)
    Y = copy.deepcopy(y_part)
    to_testy = Y.pop(i)
    xx = np.concatenate((X[0],X[1]),axis=0)
    yy = np.concatenate((Y[0],Y[1]),axis=0)
    for j in range(7):
      xx = np.concatenate((xx,X[j+2]),axis=0)
      yy = np.concatenate((yy,Y[j+2]),axis=0)
    neighbors = KNeighborsClassifier(n_neighbors=k)
    neighbors.fit(xx,yy.ravel())
    y_predict = neighbors.predict(to_testx)
    e = test(y_predict,to_testy)
    error = error + e

  error = error/10
  return error

# method that see's which k produces least error
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
def multi_k(x,y):
  occ = [0,0,0,0]
  err = [0.0,0.0,0.0,0.0]
  k_choices = [1,3,5,7]

  for i in range(30):
    (k,error) = pick_k(x,y,k_choices)

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

  print('occurances where k[1,3,5,7] are best: '+str(occ))
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


# running k nearest
def run_knn(t,p,ti,w):
  percent = 0.8 
  (t_train,tw_train,t_test,tw_test) = select_data(t,w,percent)
  (p_train,pw_train,p_test,pw_test) = select_data(p,w,percent)
  (ti_train,tiw_train,ti_test,tiw_test) = select_data(ti,w,percent)
  
  (tk,t_train_error) = multi_k(t_train,tw_train)
  (pk,p_train_error) = multi_k(p_train,pw_train)
  (tik,ti_train_error) = multi_k(ti_train,tiw_train)
 
  t_neighbors = KNeighborsClassifier(n_neighbors=tk)
  t_neighbors.fit(t_train,tw_train.ravel())
  t_predict = t_neighbors.predict(t_test)
  t_error = test(t_predict,tw_test)
   
  p_neighbors = KNeighborsClassifier(n_neighbors=pk)
  p_neighbors.fit(p_train,pw_train.ravel())
  p_predict = p_neighbors.predict(p_test)
  p_error = test(p_predict,pw_test)

  ti_neighbors = KNeighborsClassifier(n_neighbors=tik)
  ti_neighbors.fit(ti_train,tiw_train.ravel())
  ti_predict = ti_neighbors.predict(ti_test)
  ti_error = test(ti_predict,tiw_test)

  return (tk,t_error,pk,p_error,tik,ti_error)






























