from copy import deepcopy as copy
import numpy as np

def all_teams(k):
    teams=[]
    for i in range(k+1):
        for j in range(k-i+1):
            team=[0]*i+[1]*j+[2]*(k-j-i)
            teams.append(team)
            #print(team)
    return teams

def helper(t,k,n):
    if k==-1:
        return [t]
    lst=[]
    for i in range(n):
        if t[k+1]<=i:
            t[k]=i
            lst+=helper(copy(t),k-1,n)
    return lst



def a2(k,n):
    t=[0]*k
    lst=[]
    for i in range(n):
        t[k-1]=i
        lst+=helper(copy(t),k-2,n)
    return lst



#for i in range(2,16):
#    print(i,len(all_teams(i)),len(a2(i,5)))
teams=a2(16,5)
print(len(teams))
idxs=np.arange(len(teams))
np.random.shuffle(idxs)
idxs=idxs[:10]
print(idxs)
print(teams[idxs])
#print (all_teams(3))
#for k in a2(4,):
#    print(k)