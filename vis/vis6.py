import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from teaming import logger
X=[i*100/(10000*16) for i in range(101)]
#frqs=[1,10,100,1000,10000]
#frqs=[1,2,3,4,6,8,12,16]
frqs=[1,2,3,4,5,6,7,8]
letter="qq"
data=[]
err=[]


log = logger.logger()
log.load("tests/evo38-5.pkl")
#log.load("tests/vary/8-3-6.pkl")
p=log.pull("position")
t=log.pull("types")
tst=log.pull("test")
print(tst)

tst=np.array(tst)
Tst=np.average(tst,axis=1)
tst=tst[-1]
print(tst)


poi=log.pull("poi")[0]
print(poi)

idx=3
nagents=len(t[0][0])
pos=p

custom_lines = [Line2D([0], [0], color="k", lw=2),
                Line2D([0], [0], color="c", lw=2),
                Line2D([0], [0], color="y", lw=2)]

for idx in [20]:#range(50):
#for idx in [40]:#range(50):
    #plt.ion()
    plt.clf()
    #plt.subplot(1,2,1)
    VALS=[0.1,0.1,0.5,0.3,0.0,0.0]
    LOL=np.array([0,0,0,1,1,1])==1.0
    txt=[str(i) for i in VALS]
    vals=np.array(VALS)*0+1000
    
    for i in range(len(txt)):
        plt.text(poi[i,0]+3,poi[i,1]+2,txt[i])
    typ=t[0][idx]
    print(typ,tst[idx])
    for i in range(nagents):
        data=[]
        for j in range(len(pos)):
            #print(np.array(pos).shape)
            p=pos[j][idx][i]
            data.append(p)
        data=np.array(data).T
        x,y=data
        tt=typ[i]
        color=['k','c','y','m','y'][tt]
        plt.plot(x,y,color,linewidth=2.5)
    plt.scatter(poi[:,0][LOL],poi[:,1][LOL],s=vals[LOL],c='#0000ff',marker="v",zorder=10000)
    LOL=LOL==0
    print(LOL)
    plt.scatter(poi[:,0][LOL],poi[:,1][LOL],s=vals[LOL],c='#ff0000',marker="^",zorder=10000)
    #plt.title(str(idx)+':'+str(tst[idx]))
    plt.legend(custom_lines, ['Agent A', 'Agent B', 'Agent C'])
    if tst[idx]<0.6:
        continue
    #plt.subplot(1,2,2)
    #plt.plot(Tst)
    plt.axes().set_aspect('equal', 'datalim')
    #plt.pause(1.0)
plt.show()