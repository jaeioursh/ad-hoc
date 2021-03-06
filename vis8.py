import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import comb
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from teaming import logger


#schedule = ["evo"+num,"base"+num,"EVO"+num]
#schedule = ["base"+num+"_"+str(q) for q in [0.0,0.25,0.5,0.75,1.0]]
AGENTS=5
ROBOTS=3
vals=sorted([0.1, 0.2, 0.4,0.3, 0.1, 0.0],reverse=True)
lbls={0:"D Rand.",1:"Approx",2:"D Avg.",3:"G",4:"D*"}
max_val=sum(vals[:ROBOTS])
for q in [4,2,0]:
    T=[]
    R=[]
    print(q)
    for i in range(4):
        log = logger.logger()
        
        log.load("tests/vary/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl")
       
        r=log.pull("reward")
        #L=log.pull("loss")
        t=log.pull("test")
        #print(t)
        r=np.array(r)

        t=np.array(t)

        
        print(np.round(t[-1,:],2))
        N=len(np.average(t,axis=0))
        t=np.average(t,axis=1)
        
        R.append(r)
        T.append(t)
    
    BEST=np.max(T,axis=0)
    R=np.mean(R,axis=0)
    std=np.std(T,axis=0)/np.sqrt(4)
    T=np.mean(T,axis=0)
    X=[i*50 for i in range(len(T))]
    #plt.subplot(2,1,1)
    #plt.plot(BEST)
    #plt.subplot(2,1,2)
    plt.plot(X,T,label=lbls[q])
    plt.fill_between(X,T-std,T+std,alpha=0.35, label='_nolegend_')

    plt.ylim([0,1.15])
    plt.grid(True)
#plt.plot(X,[0.5]*101,"--")
#plt.plot(X,[0.8]*101,"--")
#plt.legend(["Random Teaming + Types","Unique Learners","Types Only","Max single POI reward","Max reward"])
plt.xlabel("Generation")
plt.title(str(ROBOTS)+" Robots, "+str(AGENTS)+" Agents")
leg=plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
#leg=plt.legend(["Min","First Quartile","Median","Third Quartile","Max"])
#for legobj in leg.legendHandles:
#    legobj.set_linewidth(5.0)
plt.ylabel("Average Score Across "+str(N)+" Teams")
print(len(T))
plt.plot([0,X[-1]],[max_val,max_val],"--")
'''
if num[1]=="5":
    plt.title("5 agents, coupling req. of 2")
if num[1]=="8":
    plt.title("8 agents, coupling req. of 3")
'''
#plt.title("Team Performance Across \n Quartile Selection Methods")

plt.tight_layout()
#plt.savefig("figsv3/vis8_"+str(ROBOTS)+"_"+str(AGENTS)+".png")
plt.show()