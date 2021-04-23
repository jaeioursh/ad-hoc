import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import mpltern
from mpltern.ternary.datasets import get_shanon_entropies
t, l, r, v = get_shanon_entropies()
print (t)
from teaming import logger

data=[]
err=[]
def rotate(v,n):
    idxs=[[] for i in range(n)]
    count=0
    for i in range(n):
        for j in range(i,n):
            idxs[j].append(count)
            count+=1
    idxs.reverse()
    idxs=[i for idx in idxs for i in idx]
    #idxs=np.array(idxs)
    return v[idxs]

def flip(v,n):
    idxs=[[] for i in range(n)]
    count=0
    for i in range(n):
        for j in range(n-i):
            idxs[i].append(count)
            count+=1
    for i in range(n):
        idxs[i].reverse()
    idxs=[i for idx in idxs for i in idx]
    #idxs=np.array(idxs)
    print(idxs)
    return v[idxs]
offset=0
NUM="18"
for i in range(8):
    log = logger.logger()
    #log.load("tests/jj100-7.pkl")
    #log.load("tests/jj101-"+str(i)+".pkl")
    log.load("tests/evo"+NUM+"-"+str(i+offset)+".pkl")
    t=log.pull("types")
    tst=log.pull("test")
    #print(tst)

    tst=np.array(tst)
    

    avgs=np.mean(tst,axis=-1)
    print("AVG",avgs)
    nagents=len(t[0][0])
    BEST=-1#np.argmax(avgs)
    x=[]
    y=[]
    for idx in range(len(t[0])):

        typ=t[0][idx]
        #print(typ)
        count=np.bincount(typ,minlength=3)
        count=count.astype(float)
        x.append(count/nagents)
        y.append(tst[BEST][idx])

    y=np.array(y)
    print("BEST:",i," ",np.mean(y))
    if 0:
        vals=log.pull("poi vals")[0]
        print(vals)
        vals=sorted(vals,reverse=True)
        y/=(vals[0]+vals[1])
    else:
        y/=0.8
        #y/=0.8
        pass
    #rint(x)

    x=np.array(x).T
    t,l,r=x
    vt,vl,vr=np.sum(y*t),np.sum(y*l),np.sum(y*r)
    print(vt,vl,vr)
    if vl==max(vt,vl,vr):
        y=rotate(y,nagents+1)
        y=rotate(y,nagents+1)
    if vr==max(vt,vl,vr):
        y=rotate(y,nagents+1)

    
    vt,vl,vr=np.sum(y*t),np.sum(y*l),np.sum(y*r)
    if vl<vr:
        y=flip(y,nagents+1)
    vt,vl,vr=np.sum(y*t),np.sum(y*l),np.sum(y*r)
    print(vt,vl,vr)
    data.append(y)
    
y=np.mean(data,axis=0)
#y=np.linspace(0,1,num=len(y))
#y=rotate(y,nagents+1)
v=y
fig=plt.figure()
ax = plt.subplot(projection='ternary')

position="tick1"

ax.set_tlabel('Number of Agent Type A')
ax.set_llabel('Number of Agent Type B')
ax.set_rlabel('Number of Agent Type C')

ax.taxis.set_label_position(position)
ax.laxis.set_label_position(position)
ax.raxis.set_label_position(position)

vilue=[i for i in range(9)]
ax.taxis.set_ticks(vilue)
ax.laxis.set_ticks(vilue)
ax.raxis.set_ticks(vilue)
'''
ax.set_ternary_lim(
    0.0, 8.0,  # tmin, tmax
    0.0, 8.0,  # lmin, lmax
    0.0, 8.0,  # rmin, rmax
)
'''
ax.set_tlim(0.0,8.0)
ax.set_rlim(0.0,8.0)
ax.set_llim(0.0,8.0)
if nagents==8:
    SIZE=1000
else:
    SIZE=2000
#t,l,r=np.array(t)*8.0,np.array(l)*8.0,np.array(r)*8.0
print(t)
cs=ax.scatter(t, l, r, c=v,s=SIZE)#,vmin=0,vmax=0.8)
#print(t)
#print(v)
cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)
colorbar.set_label('Team Fitness')
cs.set_clim(0.0, 1.00)
#plt.title("Random POI Positions and Values\n")
#plt.title("Fixed POI Positions and Values\n")
#plt.title("Ability Types\n")
plt.grid()
plt.tight_layout()

plt.savefig("figsv2/Figure_"+NUM+".pdf")
plt.show()

