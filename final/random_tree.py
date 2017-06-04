from data import *
from sklearn.ensemble import RandomForestClassifier

# cross validation
# returns error with associated number of estimators
def run_cross(x,y,estimators):
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
    forest = RandomForestClassifier(n_estimators=estimators)
    forest.fit(xx,yy.ravel())
    y_predict = forest.predict(to_testx)
    e = test(y_predict,to_testy)
    error = error + e 

  error = error/10
  return error

# method that see's which number of estimators produces least error
def pick_estimator(x_train,y_train,e_choices):
  error = 1 
  estimator = e_choices[0]
  iterate = len(e_choices)
  for i in range(iterate):
    e = run_cross(x_train,y_train,e_choices[i])
#    print('k = '+str(k_choices[i])+', error: '+str(e))
    if (e < error):
      error = e
      estimator = e_choices[i]
  return (estimator,error)

# shuffling data and running many times
def multi_e(x,y):
  occ = [0,0,0,0]
  err = [0,0,0,0,0.0,0.0]
  e_choices = [16,32,64,128]

  for i in range(10):
    (estimator,error) = pick_estimator(x,y,e_choices)

    if(estimator==16):
      occ[0] = occ[0]+1
      err[0] = err[0]+error
    if(estimator==32):
      occ[1] = occ[1]+1
      err[1] = err[1]+error
    if(estimator==64):
      occ[2] = occ[2]+1
      err[2] = err[2]+error
    if(estimator==128):
      occ[3] = occ[3]+1
      err[3] = err[3]+error

  print('occurances where estimators[16,32,64,128] are best: '+str(occ))
  i = occ.index(max(occ))
  final_error = err[i]/occ[i]
  estimator = 16
  if(i==1):
    estimator = 32
  if(i==2):
    estimator = 64
  if(i==3):
    estimator = 128
  return(estimator,final_error)


# running random forest
def run_random(t,p,ti,w):
  percent = 0.8
  (t_train,tw_train,t_test,tw_test) = select_data(t,w,percent)
  (p_train,pw_train,p_test,pw_test) = select_data(p,w,percent)
  (ti_train,tiw_train,ti_test,tiw_test) = select_data(ti,w,percent)

  print('\nTeam')
  (te,t_train_error) = multi_e(t_train,tw_train)
  print('Player')
  (pe,p_train_error) = multi_e(p_train,pw_train)
  print('Team Isolated')
  (tie,ti_train_error) = multi_e(ti_train,tiw_train)

  t_neighbors = RandomForestClassifier(n_estimators=te)
  t_neighbors.fit(t_train,tw_train.ravel())
  t_predict = t_neighbors.predict(t_test)
  t_error = test(t_predict,tw_test)

  p_neighbors = RandomForestClassifier(n_estimators=pe)
  p_neighbors.fit(p_train,pw_train.ravel())
  p_predict = p_neighbors.predict(p_test)
  p_error = test(p_predict,pw_test)

  ti_neighbors = RandomForestClassifier(n_estimators=tie) 
  ti_neighbors.fit(ti_train,tiw_train.ravel())
  ti_predict = ti_neighbors.predict(ti_test)
  ti_error = test(ti_predict,tiw_test)

  return (te,t_error,pe,p_error,tie,ti_error)

