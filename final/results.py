from svm import *

(t,p,ti,w) = shuffle(team,player,team_iso)
print('all team with linear kernal')
(team_C,team_error,player_C,player_error) = trial(t,p,w,1)

(pt,pp,pw) = shuffle(playoff_team,playoff_player)
print('\nplayoff teams with linear kernal')
(pteam_C,pteam_error,pplayer_C,pplayer_error) = trial(pt,pp,pw,1)
