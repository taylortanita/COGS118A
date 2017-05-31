import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import svm 
import copy
import math


# method that removes games that 'best players' didn't play in
def remove_games(team,player):
  (row,col) = player.shape
  i = 0
  while (i < row):
    if (team[i][0] != player[i][0]):
      team = np.delete(team,i,0)
    else:
      i = i + 1
  team = team[:row]
  return team

# method that selects the percentage specified for training and testing
# purposed# returns a tuple with the training and testing data
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


# shuffles data and returns 3 matrices with player data, team data and
# associated wins
def shuffle(team,player,team_iso):
  #shuffle data
  data = np.concatenate((team,player),axis=1)
  data = np.concatenate((data,team_iso),axis=1)
  np.random.shuffle(data)
  # get win/losses
  wins = data[:,[0]]
  # separate back
  team = data[:,:20]
  player = data[:,20:39]
  team_iso = data[:,39:]
  # remove date- not important metric
  # remove wins/losses
  team = np.delete(team,0,1)
  return (team,player,team_iso,wins)


###### setting up data ######
data = sio.loadmat('warriors.mat')
team = data['team']
player = data['player']
player = np.delete(player,21,1)
team = remove_games(team,player)

playoff_team = data['team']
playoff_player = data['player']
playoff_player = np.delete(playoff_player,21,1)
playoff_team = remove_games(playoff_team,playoff_player)

teams = ['nuggets.mat', 'wizards.mat', 'pacers.mat', 'jazz.mat', 'raptors.mat']
teams = teams+['spurs.mat','kings.mat','trailblazers.mat','suns.mat','76ers.mat']
teams = teams+['magic.mat','thunder.mat','knicks.mat','pelicans.mat','timberwolves.mat']
teams = teams+['bucks.mat','heat.mat','grizzlies.mat','lakers.mat','clippers.mat']
teams = teams+['rockets.mat','pistons.mat','mavericks.mat','cavaliers.mat','bulls.mat']
teams = teams+['hornets.mat','nets.mat','hawks.mat','celtics.mat']

playoff_teams = ['wizards.mat','pacers.mat','jazz.mat','raptors.mat']
playoff_teams = playoff_teams+['spurs.mat','trailblazers.mat','thunder.mat','bucks.mat']
playoff_teams = playoff_teams+['grizzlies.mat','clippers.mat','rockets.mat']
playoff_teams = playoff_teams+['cavaliers.mat','bulls.mat','hawks.mat','celtics.mat']

for i in range(29):
  data = sio.loadmat(teams[i])
  team1 = data['team']
  player1 = data['player']
  player1 = np.delete(player1,21,1)
  team1 = remove_games(team1,player1)
  team = np.concatenate((team,team1),axis=0)
  player = np.concatenate((player,player1),axis=0)

  if (teams[i] in playoff_teams):
    playoff_team1 = data['team']
    playoff_player1 = data['player']
    playoff_player1 = np.delete(playoff_player1,21,1)
    playoff_team1 = remove_games(playoff_team1,playoff_player1)
    playoff_player = np.concatenate((playoff_player,playoff_player1),axis=0)
    playoff_team = np.concatenate((playoff_team,playoff_team1),axis=0)

# when calculating field goal percentage, sometimes divide by zero.. method to get rid 
# infinite values
def remove_inf(x):
  if ((x == float("inf")) | math.isnan(x)):
    return 0.0
  else:
    return x

# isolating data
team_iso = team[:,0:2]
temp = np.subtract(team[:,2:6],player[:,2:6])
team_iso = np.concatenate((team_iso,temp),axis=1)
team_iso = team_iso[:,1:]
fg = np.divide(team_iso[:,3],team_iso[:,4])
(length,) = fg.shape
fg = np.reshape(fg,(length,1))
fg = np.apply_along_axis(remove_inf,1,fg)
if(float("inf") in fg):
  print('infinite value in fg team iso')
team_iso = np.concatenate((team_iso,fg),axis=1)
temp = np.subtract(team[:,7:9],player[:,7:9])
team_iso = np.concatenate((team_iso,temp),axis=1)
tg = np.divide(team_iso[:,6],team_iso[:,7])
(length,) = tg.shape
tg = np.reshape(tg,(length,1))
tg = np.apply_along_axis(remove_inf,1,tg)
if(float("inf") in tg):
  print('infinite value in tg team iso')
team_iso = np.concatenate((team_iso,tg),axis=1)
temp = np.subtract(team[:,10:12],player[:,10:12])
team_iso = np.concatenate((team_iso,temp),axis=1)
ft = np.divide(team_iso[:,9],team_iso[:,10])
(length,) = ft.shape
ft = np.reshape(ft,(length,1))
ft = np.apply_along_axis(remove_inf,1,ft)
if(float("inf") in ft):
  print('infinite value in ft team iso')
team_iso = np.concatenate((team_iso,ft),axis=1)
temp = np.subtract(team[:,13:21],player[:,13:21])
team_iso = np.concatenate((team_iso,temp),axis=1)

pteam_iso = playoff_team[:,0:2]
temp = np.subtract(playoff_team[:,2:6],playoff_player[:,2:6])
pteam_iso = np.concatenate((pteam_iso,temp),axis=1)
pteam_iso = pteam_iso[:,1:]
fg = np.divide(pteam_iso[:,3],pteam_iso[:,4])
(length,) = fg.shape
fg = np.reshape(fg,(length,1))
fg = np.apply_along_axis(remove_inf,1,fg)
if(float("inf") in fg):
  print('infinite value in fg playoff team iso')
pteam_iso = np.concatenate((pteam_iso,fg),axis=1)
temp = np.subtract(playoff_team[:,7:9],playoff_player[:,7:9])
pteam_iso = np.concatenate((pteam_iso,temp),axis=1)
tg = np.divide(pteam_iso[:,6],pteam_iso[:,7])
(length,) = tg.shape
tg = np.reshape(tg,(length,1))
tg = np.apply_along_axis(remove_inf,1,tg)
if(float("inf") in tg):
  print('infinite value in tg playoff team iso')
pteam_iso = np.concatenate((pteam_iso,tg),axis=1)
temp = np.subtract(playoff_team[:,10:12],playoff_player[:,10:12])
pteam_iso = np.concatenate((pteam_iso,temp),axis=1)
ft = np.divide(pteam_iso[:,9],pteam_iso[:,10])
(length,) = ft.shape
ft = np.reshape(ft,(length,1))
ft = np.apply_along_axis(remove_inf,1,ft)
if(float("inf") in ft):
  print('infinite value in ft playoff team iso')
pteam_iso = np.concatenate((pteam_iso,ft),axis=1)
temp = np.subtract(playoff_team[:,13:21],playoff_player[:,13:21])
pteam_iso = np.concatenate((pteam_iso,temp),axis=1)

#remove date and wins because irrelevant
player = np.delete(player,0,1)
player = np.delete(player,0,1)
team = np.delete(team,0,1)
playoff_player = np.delete(playoff_player,0,1)
playoff_player = np.delete(playoff_player,0,1)
playoff_team = np.delete(playoff_team,0,1)
team_iso = np.delete(team_iso,0,1)
pteam_iso = np.delete(pteam_iso,0,1)







