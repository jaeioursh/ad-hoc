import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from teaming import logger
num="38"
schedule = ["evo"+num,"base"+num]
#schedule = ["evo"+num,"base"+num,"EVO"+num]
#schedule = ["base"+num+"_"+str(q) for q in [0.0,0.25,0.5,0.75,1.0]]
for q in schedule:
    T=[]
    R=[]
    for i in range(8):
        log = logger.logger()
        
        log.load("tests/"+q+'-'+str(i)+".pkl")
       
        r=log.pull("reward")
        #L=log.pull("loss")
        t=log.pull("test")
        #print(t)
        r=np.array(r)

        t=np.array(t)

        if num[0]=="2":
            scale=0.64
        if num[0]=="1" or num[0]=="3":
            scale=0.8
        if num[0]=="0":
            vals=log.pull("poi vals")[0]
            print(vals)
            vals=sorted(vals,reverse=True)
            scale=(vals[0]+vals[1])

        N=len(np.average(t,axis=0))
        t=np.average(t,axis=1)/scale
        
        R.append(r)
        T.append(t)
    

    R=np.mean(R,axis=0)
    std=np.std(T,axis=0)/np.sqrt(8)
    T=np.mean(T,axis=0)
    X=[i*50 for i in range(len(T))]
    #plt.subplot(2,1,1)
    #plt.plot(R)
    #plt.subplot(2,1,2)
    plt.plot(X,T)
    plt.fill_between(X,T-std,T+std,alpha=0.35)

    plt.ylim([0,1.1])
    plt.grid(True)
#plt.plot(X,[0.5]*101,"--")
#plt.plot(X,[0.8]*101,"--")
#plt.legend(["Random Teaming + Types","Unique Learners","Types Only","Max single POI reward","Max reward"])
plt.xlabel("Generation")
plt.legend(["Ad hoc Teaming","CCEA"])
#leg=plt.legend(["Min","First Quartile","Median","Third Quartile","Max"])
#for legobj in leg.legendHandles:
#    legobj.set_linewidth(5.0)
plt.ylabel("Average Score Across "+str(N)+" Teams")
'''
if num[1]=="5":
    plt.title("5 agents, coupling req. of 2")
if num[1]=="8":
    plt.title("8 agents, coupling req. of 3")
'''
#plt.title("Team Performance Across \n Quartile Selection Methods")
plt.savefig("figsv2/F_"+schedule[0][-2:]+".pdf")
plt.tight_layout()
plt.show()