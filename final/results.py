from svm import *
from decision import *
from knn import *
from random_tree import *

(t,p,ti,w) = shuffle(team,player,team_iso)
(pt,pp,pti,pw) = shuffle(playoff_team,playoff_player,pteam_iso)


## SVM ###
print('SVM')
print('all team with linear kernal')
(team_C,team_error,player_C,player_error,iso_C,iso_error) = trial(t,p,ti,w,1)
print('team C: '+str(team_C)+', error: '+str(team_error))
print('player C: '+str(player_C)+', error: '+str(player_error))
print('team isolated C: '+str(iso_C)+', error: '+str(iso_error))

print('\nplayoff teams with linear kernal')
(pteam_C,pteam_error,pplayer_C,pplayer_error,piso_C,piso_error) = trial(pt,pp,pti,pw,1)
print('playoff team C: '+str(pteam_C)+', error: '+str(pteam_error))
print('playoff player C: '+str(pplayer_C)+', error: '+str(pplayer_error))
print('playoff team isolated C: '+str(piso_C)+', error: '+str(piso_error))


## Decision Tree ## 
print('\nDecision Tree')
print('\nall teams')
print('built in')
(D_team,team_error,D_player,player_error,D_iso,iso_error) = decisions(p,t,ti,w,0)
print('team depth: '+str(D_team)+', error: '+str(team_error))
print('player depth: '+str(D_player)+', error: '+str(player_error))
print('team isolated depth: '+str(D_iso)+', error: '+str(iso_error))
print('\nmanual')
(D_team2,team_error2,D_player2,player_error2,D_iso2,iso_error2) = decisions(p,t,ti,w,1)
print('team depth: '+str(D_team2)+', error: '+str(team_error2))
print('player depth: '+str(D_player2)+', error: '+str(player_error2))
print('team isolated depth: '+str(D_iso2)+', error: '+str(iso_error2))

print('\n\nplayoff teams')
print('built in')
(pD_team,pteam_error,pD_player,pplayer_error,pD_iso,piso_error) = decisions(pp,pt,pti,pw,0)
print('team depth: '+str(pD_team)+', error: '+str(pteam_error))
print('player depth: '+str(pD_player)+', error: '+str(pplayer_error))
print('team isolated depth: '+str(pD_iso)+', error: '+str(piso_error))
print('\nmanual')

(pD_team2,pteam_error2,pD_player2,pplayer_error2,pD_iso2,piso_error2) = decisions(pp,pt,pti,pw,1)
print('team depth: '+str(pD_team2)+', error: '+str(pteam_error2))
print('player depth: '+str(pD_player2)+', error: '+str(pplayer_error2))
print('team isolated depth: '+str(pD_iso2)+', error: '+str(piso_error2))


## K-Nearest Neighbors ##
print('\nK-Nearest Neighbors')
print('\nall teams')
(team_K,team_error,player_K,player_error,iso_K,iso_error) = run_knn(t,p,ti,w)
print('team K: '+str(team_K)+', error: '+str(team_error))
print('player K: '+str(player_K)+', error: '+str(player_error))
print('team isolated K: '+str(iso_K)+', error: '+str(iso_error))

print('\nplayoff teams')
(pteam_K,pteam_error,pplayer_K,pplayer_error,piso_K,piso_error) = run_knn(pt,pp,pti,pw)
print('team K: '+str(pteam_K)+', error: '+str(pteam_error))
print('player K: '+str(pplayer_K)+', error: '+str(pplayer_error))
print('team isolated K: '+str(piso_K)+', error: '+str(piso_error))


## Random Forests ##
print('\nRandom Forests')
print('\nall teams')
(team_E,team_error,player_E,player_error,iso_E,iso_error) = run_random(t,p,ti,w)
print('team number of trees: '+str(team_E)+', error: '+str(team_error))
print('player numer of trees: '+str(player_E)+', error: '+str(player_error))
print('team isolated number of trees: '+str(iso_E)+', error: '+str(iso_error))

print('\nplayoff teams')
(pteam_K,pteam_error,pplayer_K,pplayer_error,piso_K,piso_error) = run_knn(pt,pp,pti,pw)
print('team K: '+str(pteam_K)+', error: '+str(pteam_error))
print('player K: '+str(pplayer_K)+', error: '+str(pplayer_error))
print('team isolated K: '+str(piso_K)+', error: '+str(piso_error))




