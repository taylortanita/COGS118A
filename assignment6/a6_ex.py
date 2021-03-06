from a6 import *

data = sio.loadmat('ionosphere.mat')
x = data['X']
y = data['Y']
y = convert_array(y)

##### Q3 #####
print('Q3: DECISION TREE')
### 80/20 split
percent = 0.8
print('For 80/20 split')
(x_train,y_train,x_test,y_test) = select_data(x,y,percent)
D_choices = [1,2,4,8,16,32,64,128,256,400]

# getting appropriate D value to use
(D,training_error) = pick_D(x_train,y_train,D_choices)
if (D==400):
  print('Optimal D: default')
else:
  print('Optimal D: '+str(D))
print('Validation Error: '+str(training_error))

clf = tree.DecisionTreeClassifier(max_depth=D)
if (D == 400):
  clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
predict = clf.predict(x_train)
training_error = test(predict,y_train)
print('Training Error: '+str(training_error))

# Testing
clf = tree.DecisionTreeClassifier(max_depth=D)
if (D == 400):
  clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
predict = clf.predict(x_test)
testing_error = test(predict,y_test)
print('Testing Error: '+str(testing_error)+'\n')


### 60/40 split
percent = 0.6 
print('For 60/40 split')
(x_train,y_train,x_test,y_test) = select_data(x,y,percent)
D_choices = [1,2,4,8,16,32,64,128,256,400]

# getting appropriate D value to use
(D,training_error) = pick_D(x_train,y_train,D_choices)
if (D==400):
  print('Optimal D: default')
else:
  print('Optimal D: '+str(D))
print('Validation Error: '+str(training_error))

clf = tree.DecisionTreeClassifier(max_depth=D)
if (D == 400):
  clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
predict = clf.predict(x_train)
training_error = test(predict,y_train)
print('Training Error: '+str(training_error))

# Testing
clf = tree.DecisionTreeClassifier(max_depth=D)
if (D == 400):
  clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
predict = clf.predict(x_test)
testing_error = test(predict,y_test)
print('Testing Error: '+str(testing_error)+'\n')


###### Q4 #####
print('Q4: k-NEAREST NEIGHBORS')

percent = 0.8
(x_train,y_train,x_test,y_test) = select_data(x,y,percent)
print('For 80/20 split')
k_choices = [1,3,5,7]
(k,training_error) = multi_k(x_train,y_train,percent,k_choices)
print('Optimal k: '+str(k))
print('Validation Error: '+str(training_error))
e_train = run_k(x_train,y_train,k)
print('Training Error: '+str(e_train))
e = run_k(x_test,y_test,k)
print('Testing Error: '+str(e)+'\n')


percent = 0.6
(x_train,y_train,x_test,y_test) = select_data(x,y,percent)
print('For 60/40 split')
k_choices = [1,3,5,7]
(k,training_error) = multi_k(x_train,y_train,percent,k_choices)
print('Optimal k: '+str(k))
print('Validation Error: '+str(training_error))
e_train = run_k(x_train,y_train,k)
print('Training Error: '+str(e_train))
e = run_k(x_test,y_test,k)
print('Testing Error: '+str(e)+'\n')

