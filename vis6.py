import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from teaming import logger
data=[]
err=[]
AGENTS=4
ROBOTS=3
ROWS=2
COLS=2

i=1

q=4
fname="tests/vary/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl"

log = logger.logger()
#log.load("tests/evo38-5.pkl")
log.load(fname)
p=log.pull("position")
t=log.pull("types")
tst=log.pull("test")


tst=np.array(tst)
Tst=np.average(tst,axis=1)
tst=tst[-1]



poi=log.pull("poi")[0]



nagents=len(t[0][0])
pos=p

N=len(t[0])

for idx in range(N):
    plt.subplot(ROWS,COLS,idx+1)#range(50):
    #for idx in [40]:#range(50):
    #plt.ion()
    #plt.clf()
    #plt.subplot(1,2,1)
    VALS=[0.1, 0.2, 0.4,0.3, 0.1, 0.0]
 
    txt=[str(i) for i in VALS]
    vals=np.array(VALS)*0+1000
    
    
    typ=t[0][idx]

    for i in range(nagents):
        data=[]
        for j in range(len(pos)):
            #print(np.array(pos).shape)
            p=pos[j][idx][i]
            data.append(p)
        data=np.array(data).T
        x,y=data
        tt=typ[i]
        color=[".",",","*","v","^","<",">","1","2","3","4","8"][tt]
        plt.plot(x,y,color="k",marker=color,linewidth=1.0)

    lgnd=["Policy "+str(i) for i in typ]
    #print(lgnd)
    plt.legend(lgnd)

    for i in range(len(txt)):
        plt.text(poi[i,0]+3,poi[i,1]+2,txt[i])

    plt.scatter(poi[:,0],poi[:,1],c='#0000ff',marker="v",zorder=10000)


    #plt.title(str(idx)+':'+str(tst[idx]))
    
    plt.title("score: "+str(round(tst[idx],3)))
    #if tst[idx]<0.6:
    #    continue
    #plt.subplot(1,2,2)
    #plt.plot(Tst)
    #plt.axes().set_aspect('equal', 'datalim')
    #plt.pause(1.0)
plt.show()