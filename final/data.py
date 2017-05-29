import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import svm 
import copy
import math

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

data = sio.loadmat('warriors.mat')
team = data['team']
player = data['player']
player = np.delete(player,21,1)
team = remove_games(team,player)

data = sio.loadmat('nuggets.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('wizards.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('jazz.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('raptors.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('spurs.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('kings.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('trailblazers.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('suns.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('76ers.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('magic.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('thunder.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('knicks.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('pelicans.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('timberwolves.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('bucks.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('heat.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('grizzlies.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('lakers.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('clippers.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('rockets.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('pistons.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('mavericks.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('cavaliers.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('bulls.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('hornets.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('nets.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('hawks.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

data = sio.loadmat('celtics.mat')
team1 = data['team']
player1 = data['player']
player1 = np.delete(player1,21,1)
team1 = remove_games(team1,player1)
team = np.concatenate((team,team1),axis=0)
player = np.concatenate((player,player1),axis=0)

#remove date and wins because irrelevant
player = np.delete(player,0,1)
player = np.delete(player,0,1)
team = np.delete(team,0,1)

print(player.shape)
print(team.shape)

#shuffle data
data = np.concatenate((team,player),axis=1)
np.random.shuffle(data)

# get win/losses
wins = data[:,[0]]

# separate back
team = data[:,:20]
player = data[:,20:]


# remove date- not important metric
# remove wins/losses
team = np.delete(team,0,1)
