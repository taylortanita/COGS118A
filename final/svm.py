from data import *
import operator

# method to determine the error when using a specific C value and svm type
# returns error 
# svm_type: 1 = linear, 0 = RBF
def test_c(x,y,c,svm_type):
# in order for split to work, needs to be a multiple of what you are splitting
# it to, so take the first partition will contain the remainder data
  (size,width) = x.shape
  mod = size % 10
  xx = x[mod:]
  yy = y[mod:]
  x_part = np.split(xx,10)
  y_part = np.split(yy,10)
  x_part[0] = np.concatenate((x_part[0],x[:mod]),axis=0)
  y_part[0] = np.concatenate((y_part[0],y[:mod]),axis=0)
# 10-fold cross validation
  error = 0
# use appropriate svm type
  clf = svm.LinearSVC(C=c)
  if (svm_type == 0):
    clf = svm.SVC(C=c)
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
#   fitting data
    clf.fit(xx,yy)
    predict = clf.predict(to_testx)
    e = test(predict,to_testy)
    error = error + e
# calculating crossvalidation error
  crossval_error = error/10
  return crossval_error



# binary search method
def binary(c1,c2,x,y,svm_type):
# calculate errors for endpoints and mid value
  mid = (c1 + c2)/2.0
  err_1 = test_c(x,y,c1,svm_type)
  err_mid = test_c(x,y,mid,svm_type)
  err_2 = test_c(x,y,c2,svm_type)
# threshold
  if (abs(c2 - c1) < .0001):
    return (c1,err_1)
# picking which endpoint to compare middle value to 
  c_val = c1
  error = 0
  if (err_1 < err_2):
    c_val = c1
    error = err_1
  else:
    c_val = c2
    error = err_2
# if the middle value has a lower error, continue searching, if not, return the
# current C value and error
  if ((err_mid < err_1) | (err_mid < err_2)):
    if (c_val < err_mid):
      return binary(c_val,mid,x,y,svm_type)
    else:
      return binary(mid,c_val,x,y,svm_type)
  else:
    return (c_val,error)



# method to pick the best C
# tries 4 different C values 
# returns the best C value and the error associated with it
def pick_C(x,y,svm_type):
# test the different C values
  error = test_c(x,y,10,svm_type)
  err_100 = test_c(x,y,100,svm_type)
  err_1 = test_c(x,y,1,svm_type)
  err_p1 = test_c(x,y,.1,svm_type)
  err_01 = test_c(x,y,.01,svm_type)
  err_001 = test_c(x,y,.001,svm_type)
# put results in a list to extract the minimum
  values=[(err_100,100),(error,10),(err_1,1),(err_p1,.1)]
  values = values + [(err_01,.01),(err_001,.001)]
  (error,c) = min(values,key=min_func)
# getting correct interval of c's
  c1 = 0
  c2 = 0
# if one of the edge values (i.e., 100 or .001)
  if (c == 100):
    c1 = 10
    c2 = 100
  elif (c == .001):
    c1 = .001
    c2 = .01
# if c if one of the middle values (i.e., 10,1,.1,.01)
  for i in range(1,5):
    (vv,cc) = values[i]
    if (c == cc):
      (v1,cc1) = values[i-1]
      (v2,cc2) = values[i+1]
      if (v1 < v2):
        c1 = cc
        c2 = cc1
      else:
        c1 = cc2
        c2 = cc
# generated minimum error
  (c,error) = binary(c1,c2,x,y,svm_type)
  return (c,error)



# function to settle ties in errors to ensure lowest C is used
def min_func((error,c)):
# weigh the error significantly more so that the only difference will occur in a
# tie
  score = error * 1000
  if (c == 100):
    score = score - .000001
  if (c == 10):
    score = score - .000002
  if (c == 1):
    score = score - .000003
  if (c == .1):
    score = score - .000004
  if (c == .01):
    score = score - .000005
  if (c == .001):
    score = score - .000006
  return score


# shuffles data and runs multiple times
# reports testing error
def trial(team,player,iso,wins,svm_type):
  TC = {}
  TE = {} 
  PC = {}
  PE = {} 
  IC = {}
  IE = {} 
  for i in range(10):
    (t_train,tw_train,t_test,tw_test) = select_data(team,wins,.8)
    (p_train,pw_train,p_test,pw_test) = select_data(player,wins,.8)
    (i_train,iw_train,i_test,iw_test) = select_data(iso,wins,.8)
    (tc,t_error) = pick_C(t_train,tw_train,svm_type)
    (pc,p_error) = pick_C(p_train,pw_train,svm_type)
    (ic,i_error) = pick_C(i_train,iw_train,svm_type)
    
    print('team c: '+str(tc)+', error: '+str(t_error))    
    print('player c: '+str(pc)+', error: '+str(p_error))    
    print('team isolated c: '+str(ic)+', error: '+str(i_error)+'\n')    

    if (TC.has_key(tc)):
      TC[tc] = TC[tc]+1 
    else:
      TC[tc] = 1
    if (PC.has_key(pc)):
      PC[pc] = PC[pc]+1 
    else:
      PC[pc] = 1
    if (IC.has_key(ic)):
      IC[ic] = IC[ic]+1 
    else:
      IC[ic] = 1
    
    team_clf = svm.LinearSVC(C=tc)
    if (svm_type == 0):
      team_clf = svm.SVC(C=tc)
    team_clf.fit(t_train,tw_train)
    t_predict = team_clf.predict(t_test)
    t_error = test(t_predict,tw_test)
    if (TE.has_key(tc)):
      TE[tc] = TE[tc]+t_error 
    else:
      TE[tc] = t_error
    
    player_clf = svm.LinearSVC(C=pc)
    if (svm_type == 0):
      player_clf = svm.SVC(C=pc)
    player_clf.fit(p_train,pw_train)
    p_predict = team_clf.predict(p_test)
    p_error = test(p_predict,pw_test)
    if (PE.has_key(pc)):
      PE[pc] = PE[pc]+p_error 
    else:
      PE[pc] = p_error

     
    iso_clf = svm.LinearSVC(C=ic)
    if (svm_type == 0):
      iso_clf = svm.SVC(C=ic)
    iso_clf.fit(i_train,iw_train)
    i_predict = iso_clf.predict(i_test)
    i_error = test(i_predict,iw_test)
    if (IE.has_key(ic)):
      IE[ic] = IE[ic]+i_error 
    else:
      IE[ic] = i_error
 
  team_C = max(TC.iteritems(), key=operator.itemgetter(1))[0]
  player_C = max(PC.iteritems(), key=operator.itemgetter(1))[0]
  iso_C = max(IC.iteritems(), key=operator.itemgetter(1))[0]
  team_error = TE[team_C]
  player_error = PE[player_C]
  iso_error = IE[iso_C]
  team_error = team_error/TC[team_C]
  player_error = player_error/PC[player_C]  
  iso_error = iso_error/IC[iso_C]  
  print('\n')
  print(TC)
  print(TE)
  print(PC)
  print(PE)
  print(IC)
  print(IE)
  print('\n')
  return (team_C,team_error,player_C,player_error,iso_C,iso_error)
























