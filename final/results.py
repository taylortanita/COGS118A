from svm import *

(t,p,ti,w) = shuffle(team,player,team_iso)
print('all team with linear kernal')
(team_C,team_error,player_C,player_error,iso_C,iso_error) = trial(t,p,ti,w,1)

(pt,pp,pti,pw) = shuffle(playoff_team,playoff_player,pteam_iso)
print('\nplayoff teams with linear kernal')
(pteam_C,pteam_error,pplayer_C,pplayer_error,piso_C,piso_error) = trial(pt,pp,pti,pw,1)
