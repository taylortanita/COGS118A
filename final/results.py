from svm import *
from decision import *

(t,p,ti,w) = shuffle(team,player,team_iso)
(pt,pp,pti,pw) = shuffle(playoff_team,playoff_player,pteam_iso)

## SVM ###
'''
print('SVM')
print('all team with linear kernal')
(team_C,team_error,player_C,player_error,iso_C,iso_error) = trial(t,p,ti,w,1)

print('\nplayoff teams with linear kernal')
(pteam_C,pteam_error,pplayer_C,pplayer_error,piso_C,piso_error) = trial(pt,pp,pti,pw,1)
'''

print('\nDecision Tree')
print('all teams')
(D_team,team_errr,D_player,player_error,D_team_iso,team_iso_error) = decisions(p,t,ti,w)







