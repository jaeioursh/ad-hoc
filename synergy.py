
from random import shuffle
from itertools import combinations,permutations
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import pickle
import multiprocessing as mp
import time
from vis2.analysis1 import vis

class SimpleRoverDomain:
    def __init__(self,N=6,T=4,couple=2,poi=np.array([1,2,1,4,0])):
        self.N=N #number of agents
        self.T=T #team size
        self.couple=couple
        seed=np.random.get_state()
        #print("SEED",seed)
        np.random.seed(0)
        self.capabilities=np.random.random((N,len(poi)))
        self.capabilities=np.round(self.capabilities,2)
        print(self.capabilities)
        np.random.set_state(seed)
        #self.capabilities[:]=1
        self.poi=poi

    def G(self,act,team):
        poi=self.poi.copy()
        poi=poi.astype(float)
        #print(len(poi))
        count=np.bincount(act,minlength=len(poi))
        for a,t in zip(act,team):
            poi[a]*=self.capabilities[t,a]
        
        return np.sum(poi[count>=self.couple])

    def D(self,act,team):
        act=np.array(act)
        gg=self.G(act,team)
        d1=np.zeros(len(act))
        for i in range(len(act)):
            ACT=act.copy()
            ACT[i]=len(self.poi)-1
            g=self.G(ACT,team)
            d1[i]=gg-g 

        return d1
    
    def gen_teams(self,n_teams):
        teams=[]
        if not n_teams:
            #print(comb(self.N,self.T))
            for t in combinations(range(self.N),self.T):
                teams.append(list(t))
        else:
            for i in range(n_teams):
                teams.append(np.sort(np.random.choice(self.N,self.T,replace=False)))
        return teams

def action(graph,team): #feed forward/nohidden layer
    graph=graph[team,:,:]
    graph=graph[:,team,:]
    ACT=np.sum(graph,axis=1)#potentially find the largest cycle
    ACT=np.argmax(ACT,axis=1)
    return ACT

def vaction(graph,team): #voting
    votes=np.argmax(graph,axis=2)
    ACT=[]
    for t in team:
        team2=[i for i in team if i!=t]
        ACT.append(np.argmax(np.bincount(votes[t,team])))
    return np.array(ACT)

def permute(n,k):
    lst=[np.zeros(k,int)]
    while not np.all(lst[-1]==np.zeros(k,int)+n-1):
        nxt=lst[-1].copy()
        for i in range(k):
            nxt[i]+=1
            if nxt[i]==n:
                nxt[i]=0
            else:
                break
        lst.append(nxt)
        print(nxt)
    return lst

    



class optim:
    def __init__(self,N,T,C):
        self.env=SimpleRoverDomain(N,T,C)
        self.N=N
        self.T=T
        self.C=C

    def search(self):
        teams=self.env.gen_teams(0)
        acts=[]
        perms=permute(len(self.env.poi)-1,self.T)
        print(perms)
        g=0
        vex=[]
        for t in teams:
            scores=[]
            for p in perms:
                scores.append(self.env.G(p,t))
            acts.append(perms[np.argmax(scores)])
            g+=max(scores)
            vec=np.zeros(self.N,int)
            vec[t]=acts[-1]+1
            print(vec)
            vex.append(vec)
        print(g)
        vis(vex,self.env.capabilities,self.env.poi)





class RL:
    def __init__(self,env,act_fn):
        self.tag="rl"
        self.env=env
        self.N=env.N
        self.T=env.T
        self.graph=np.random.normal(0.0,0.01,size=(self.N,self.N,len(self.env.poi)-1))
        self.lr=0.01
        self.g=0
        self.chance=0.40
        self.decay=0.999
        self.act_fn=act_fn
    
    
    def train(self):
        self.g=0
        teams=self.env.gen_teams(30)
        self.chance*=self.decay
        for t in teams:
            act=self.act_fn(self.graph,t)
            rand_act=np.random.randint(0,len(self.env.poi)-1,len(act))
            rand=np.random.random(len(act))
            act[rand<self.chance]=rand_act[rand<self.chance]
            g=self.env.G(act,t)
            self.g+=g
            d=self.env.D(act,t)
            for i in range(len(t)):
                self.graph[t[i],t,act[i]]*=(1-self.lr)
                self.graph[t[i],t,act[i]]+=self.lr*d[i]
    def eval(self):
        g=0
        teams=self.env.gen_teams(0)
        acts=[]
        for t in teams:
            act=self.act_fn(self.graph,t)
            g+=self.env.G(act,t)
            acts.append(act)
        return g,acts,teams


        
        



class ccea:
    def __init__(self,env,act_fn,pop_size=32):
        self.tag="ea"
        self.N=env.N
        self.T=env.T
        self.A=len(env.poi)-1
        self.pop_size=pop_size
        self.env=env
        self.act_fn=act_fn
        self.chance=1
        self.mut_prob=0.3
        self.mut_rate=0.1
        self.pop=[[self.gen() for i in range(self.pop_size)] for i in range(self.N)]
        self.fit=np.zeros((self.pop_size,self.N))
        self.g=np.zeros(self.pop_size)
        self.teams=[]
        self.acts=[]

    def gen(self):
        return np.random.normal(1.0,0.05,size=(self.N,len(self.env.poi)-1))
    
    def mutate(self,x,team):
        x=x.copy()
        prob=np.random.random(x.shape)
        mut=np.random.normal(1.0,self.mut_rate,(len(team),x.shape[-1]))
        x_sample=x[team,:]
        x_sample[prob<self.mut_prob]*=mut[prob<self.mut_prob]
        return x

        


    def train(self):
        for i in range(self.N):
            shuffle(self.pop[i])

        teams=self.env.gen_teams(0)
        
        unique=np.unique(teams)
        self.teams=teams

        self.fit[:,:]=0
        self.g[:]=0
        self.acts=[[] for i in range(self.pop_size)]
        for i in range(self.pop_size):
            graph=[]
            for j in range(self.N):
                graph.append(self.pop[j][i])
            graph=np.array(graph)

            
            for t in teams:
                act=self.act_fn(graph,t)
                self.acts[i].append(act)
                g=self.env.G(act,t)
                d=self.env.D(act,t)
                #print(t,d)
                self.g[i]+=g
                self.fit[i,t]+=d

        for j in range(self.pop_size//2):
            for i in unique:
                if self.fit[j*2,i]>self.fit[j*2+1,i]:
                    self.pop[i][j*2+1]=self.mutate(self.pop[i][j*2],unique)
                else:
                    self.pop[i][j*2]=self.mutate(self.pop[i][j*2+1],unique)
    def eval(self):
        idx=np.argmax(self.g)
        return self.g[idx],self.acts[idx],self.teams
        

def exp1(itr,type,N_agents=6,T_size=4,Couple=2,FLAG=0,Episodes=2000,PLOT=0):
    np.random.seed(itr+1000)
    env=SimpleRoverDomain(N=N_agents,T=T_size,couple=Couple)
    if FLAG==1:
        env.capabilities[:]=1
    rnd=[]
    
    act_fn=vaction
    if type==1:
        learner=ccea(env,act_fn)
        Episodes=Episodes//10
    else:
        learner=RL(env,act_fn)
    gs=[]
    acts=[]
    teams=[]
    for i in range(Episodes):
        learner.train()
        if i%25==0:
            g,act,team=learner.eval()
            gs.append(g), acts.append(act), teams.append(team)
            rnd.append(learner.chance)
            print(i,g)
        
        
    print(max(gs))
    fname="_".join([learner.tag]+[str(i) for i in [FLAG,N_agents,T_size,Couple,itr]])+".pkl"
    print(fname)
    with open("tests/syn/"+fname, 'wb') as handle:
        pickle.dump([gs,acts,teams,env.poi,env.capabilities], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print(rnd)
    if PLOT:
        rnd=np.array(rnd)/max(rnd)*max(gs)
        plt.plot(gs)
        plt.plot(rnd)
        plt.show()

if 1:
    opt=optim(6,4,2)
    opt.search()

else:
    procs=[]
    for type in [0,1]:
        for i in range(8):
            itr=i
            #type=0
            N_agents=10
            T_size=6
            Couple=3
            FLAG=0
            p=mp.Process(target=exp1,args=(itr,type,N_agents,T_size,Couple,FLAG))
            p.start()
            time.sleep(0.05)
            procs.append(p)
            #p.join()
        for p in procs:
            time.sleep(0.05)
            p.join()
