from data import *
from sklearn import tree

# 
def divide_fg(x,y):
  x = np.concatenate((x,y),axis=1)
  fg_percent = x[:,4]
  avg = np.average(fg_percent)
  (row,col) = x.shape
  above = copy.deepcopy(x)
  below = copy.deepcopy(x)
  for i in range(row):
    if (fg_percent[i] < avg):
      above = np.concatenate((above,x[i].reshape(1,col)),axis=0)
    else:
      below = np.concatenate((below,x[i].reshape(1,col)),axis=0)
  above = above[row:,:]
  above = np.delete(above,4,1)
  below = below[row:,:]
  below = np.delete(below,4,1)
  above_y = above[:,(col-2)]
  below_y = below[:,(col-2)]
  above = np.delete(above,(col-2),1)
  below = np.delete(below,(col-2),1)
  return (above,above_y,below,below_y)


def predict_val(x,y,x_test,y_test,D):
  (a,ay,b,by) = divide_fg(x_test,y_test)

  (above,above_y,below,below_y) = divide_fg(x,y)
  clf1 = tree.DecisionTreeClassifier()
  if (D != (-1)):
    clf1 = tree.DecisionTreeClassifier(max_depth=D)
  clf1.fit(above,above_y)
  a_predict = clf1.predict(a)
  a_error = test(a_predict,ay)

  clf2 = tree.DecisionTreeClassifier()
  if (D != (-1)):
    clf2 = tree.DecisionTreeClassifier(max_depth=D)
  clf2.fit(below,below_y)
  b_predict = clf2.predict(b)
  b_error = test(b_predict,by)
  
  (a_rows,_) = above.shape
  (b_rows,_) = below.shape
  a_error = a_error * a_rows
  b_error = b_error * b_rows
  error = (a_error + b_error)/(a_rows + b_rows)
  return error

def decision(x,y,tree_type):
  (x_train,y_train,x_test,y_test) = select_data(x,y,.8)
  D_choices = [1,2,4,8,16,32,64,128,256,512,1024,-1]
  (D,training_error) = pick_D(x_train,y_train,D_choices,tree_type)
  if (D==(-1)):
    print('Optimal D: default')
  else:
    print('Optimal D: '+str(D))

  testing_error = 0.0
  if (tree_type == 1):
    testing_error = predict_val(x_train,y_train,x_test,y_test,D)  
  clf = tree.DecisionTreeClassifier()
  if (D != (-1)):
    clf = tree.DecisionTreeClassifier(max_depth=D)
  if (tree_type == 0):  
    clf.fit(x_train,y_train)
    predict = clf.predict(x_test)
    testing_error = test(predict,y_test)
  return (D,testing_error)


# method to compute error using specific D
# reports the error associated with that D value
def compute_error(x,y,D,tree_type):
  (size,width) = x.shape
  mod = size % 10
  xx = x[mod:]
  yy = y[mod:]

  x_part = np.split(xx,10)
  y_part = np.split(yy,10)

  x_part[0] = np.concatenate((x_part[0],x[:mod]),axis=0)
  y_part[0] = np.concatenate((y_part[0],y[:mod]),axis=0)

  # 5-fold cross validation
  error = 0 
  clf = tree.DecisionTreeClassifier(max_depth=D)
  # if 400 is indicated, it means the default D was used
  if (D == (-1)):
    clf = tree.DecisionTreeClassifier()
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
    e = 0.0
    if (tree_type == 1):
      e = predict_val(xx,yy,to_testx,to_testy,D)
    else:
      clf.fit(xx,yy)
      predict = clf.predict(to_testx)
      e = test(predict,to_testy)
    error = error + e 
  error = error/5
#  print('error = '+str(error))
  return error

# runs cross validation multiple times 
def run(x_train,y_train,D,tree_type):
  error = 0.0
  for i in range(30):
    error = error + compute_error(x_train,y_train,D,tree_type)
  error = error/30
  return error


# method to pick which D is best
# takes in training data and list of D values to test out
# reports the D value
def pick_D(x_train,y_train,D_choices,tree_type):
  error = 1
  D = D_choices[0]
  iterate = len(D_choices)
  for i in range(iterate):
    e = run(x_train,y_train,D_choices[i],tree_type)
    if (e < error):
      error = e
      D = D_choices[i]
  return (D,error)


# type1 = manual
# type 2 = built in
def decisions(player,team,team_iso,wins,tree_type):
  (D_team,team_error) = decision(team,wins,tree_type)
  (D_player,player_error) = decision(player,wins,tree_type)
  (D_team_iso,team_iso_error) = decision(team_iso,wins,tree_type)
  print('Team D: '+str(D_team)+', error: '+str(team_error))
  print('Player D: '+str(D_player)+', error: '+str(player_error))
  print('Team Isolated D: '+str(D_team_iso)+', error: '+str(team_iso_error))
  return (D_team,team_error,D_player,player_error,D_team_iso,team_iso_error) 





