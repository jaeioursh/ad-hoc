import re
import numpy as np
#import tensorflow as tf
import numpy as np

from copy import deepcopy as copy
from .logger import logger
import pyximport
from .cceamtl import *
from itertools import combinations
#from math import comb
from collections import deque
from random import sample
import torch

import operator as op
from functools import reduce

def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


class Net:
    def __init__(self,hidden=100):
        learning_rate=1e-2
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,1)
        )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)

    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.model(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    

def helper(t,k,n):
    if k==-1:
        return [t]
    lst=[]
    for i in range(n):
        if t[k+1]<=i:
            t[k]=i
            lst+=helper(copy(t),k-1,n)
    return lst

def s2z(s,i):
    s=s.copy()
    row=s[i,:].copy()
    s[i,:]=0
    z=np.vstack((row,s))
    return z.flatten()

def robust_sample(data,n):
    if len(data)<n: 
        smpl=data
    else:
        smpl=sample(data,n)
    return smpl

class learner:
    def __init__(self,nagents,types,sim):
        self.log=logger()
        self.nagents=nagents
        self.hist=[deque(maxlen=10000) for i in range(types)]
        self.zero=[deque(maxlen=10000) for i in range(types)]
        self.itr=0
        self.types=types
        self.team=[self.sample()]
        self.Dapprox=[Net() for i in range(self.types)]
        print(self.Dapprox)
        self.every_team=self.many_teams()
        self.test_teams=self.every_team
        sim.data["Number of Policies"]=32
        initCcea(input_shape=8, num_outputs=2, num_units=20, num_types=types)(sim.data)
        

    def act(self,S,data,trial):
        policyCol=data["Agent Policies"]
        A=[]
        for s,pol in zip(S,policyCol):
  
            a = pol.get_action(s)*2.0
            A.append(a)
        return np.array(A)
    

    def randomize(self):
        length=len(self.every_team)
        teams=[]
        
        idx=np.random.choice(length)
        t=self.every_team[idx].copy()
        #np.random.shuffle(t)
        teams.append(t)
        self.team=teams
        #self.team=np.random.randint(0,self.types,self.nagents)
    def set_teams(self):
        
    def save(self,fname="log.pkl"):
        print("saved")
        self.log.save(fname)
        netinfo={i:self.Dapprox[i].model.state_dict() for i in range(len(self.Dapprox))}
        torch.save(netinfo,fname+".mdl")

    #train_flag=0 - D
    #train_flag=1 - Neural Net Approx of D
    #train_flag=2 - Approx, one at a time
    #train_flag=3 - G
    #train_flag=4 - D*
    def run(self,env,train_flag):
        populationSize=len(env.data['Agent Populations'][0])
        pop=env.data['Agent Populations']
        #team=self.team[0]
        G=[]
        if train_flag==4 or train_flag==1:
            self.team=self.every_team
        for worldIndex in range(populationSize):
            env.data["World Index"]=worldIndex
            
            #for agent_idx in range(self.types):
            
            for team in self.team:
                s = env.reset() 
                done=False 
                #assignCceaPoliciesHOF(env.data)
                assignCceaPolicies(env.data,team)
                S,A=[],[]
                while not done:
                    self.itr+=1
                    
                    action=self.act(s,env.data,0)
                    S.append(s)
                    A.append(action)
                    s, r, done, info = env.step(action)
                #S,A=[S[-1]],[A[-1]]
                pols=env.data["Agent Policies"] 
                g=env.data["Global Reward"]
                for i in range(len(s)):
                    #z=s2z(s,i)
                    #d=r[i]
                    #pols[i].D.append(d)
                    for j in range(len(S)):
                        z=[S[j][i],A[j][i],g]
                        #if d!=0:
                        self.hist[team[i]].append(z)
                        #else:
                        #    self.zero[team[i]].append(z)
                    pols[i].Z.append(S[-1][i])
                        
                G.append(g)
            
        if train_flag!=4 and train_flag!=1:
            train_set=self.team[0]
        else:
            train_set=[i for i in range(self.types)]

        if train_flag==1 or train_flag==2:
            self.updateD(env)
            
        for t in train_set:
            #if train_flag==1:
            #    S_sample=self.state_sample(t)

            for p in pop[t]:
                
                d=p.D[-1]
                if train_flag==4:
                    p.fitness=np.sum(p.D)
                    p.D=[]
                if train_flag==3:
                    p.fitness=g

                if train_flag==1 or train_flag==2:
                    #self.approx(p,t,S_sample)
                    p.D=list(self.Dapprox[t].feed(np.array(p.Z)))
                    p.fitness=np.sum(p.D)
                    #print(p.fitness)
                    p.Z=[]
                    
                if train_flag==0:
                    p.fitness=d
        

        evolveCceaPolicies(env.data,train_set)

        self.log.store("reward",max(G))      
        return max(G)


    def updateD(self,env):
        
        pop=env.data['Agent Populations']
        populationSize=len(pop[0])
        team=self.team[0]
        for q in range(50):
            for i in team:
                S,A,D=[],[],[]
                SAD=robust_sample(self.hist[i],100)
                #SAD+=robust_sample(self.zero[i],100)
                for samp in SAD:
                    S.append(samp[0])
                    A.append(samp[1])
                    D.append([samp[2]])
                S,A,D=np.array(S),np.array(A),np.array(D)
                Z=S#np.hstack((S,A))
                self.Dapprox[i].train(Z,D)
    def state_sample(self,t):
        S=[]
        A=[]
        SAD=robust_sample(self.hist[t],100)
        if len(SAD)==0:
            SAD+=robust_sample(self.zero[t],100)
        for samp in SAD:
            s=samp[0]
            S.append(s)
        return np.array(S)

    def approx(self,p,t,S):
        
        A=[p.get_action(s) for s in S]
        A=np.array(A)
        Z=np.hstack((S,A))
        D=self.Dapprox[t].feed(Z)
        fit=np.sum(D)
        #print(fit)
        p.fitness=fit

    def put(self,key,data):
        self.log.store(key,data)


    def test(self,env,itrs=50,render=0):

        old_team=self.team
        #
        

        self.log.clear("position")
        self.log.clear("types")
        
        self.log.clear("poi")
        self.log.store("poi",np.array(env.data["Poi Positions"]))
        self.log.clear("poi vals")
        self.log.store("poi vals",np.array(env.data['Poi Static Values']))
        Rs=[]
        teams=copy(self.test_teams)
        print(teams)
        for i in range(len(teams)):

            
            
            #team=np.array(teams[i]).copy()
            #np.random.shuffle(team)
            self.team=[teams[i]]
            team=teams[i]
            #for i in range(itrs):
            assignBestCceaPolicies(env.data,team)
            #self.randomize()
            s=env.reset()
            done=False
            R=[]
            i=0
            self.log.store("types",self.team[0].copy(),i)
            
            while not done:
                
                self.log.store("position",np.array(env.data["Agent Positions"]),i)
                
                action=self.act(s,env.data,0)
                #action=self.idx2a(env,[1,1,3])
                #print(action)
                sp, r, done, info = env.step(action)
                if render:
                    env.render()
                
                s=sp
                i+=1
            g=env.data["Global Reward"]
            Rs.append(g)
        self.log.store("test",Rs)
        
        self.team=old_team

    

    def quick(self,env,episode,render=False):
        s=env.reset()
        
        for i in range(100):
            a=[[0,0] for i in range(self.nagents)]
            sp, r, done, info = env.step(a)
        return [0.0]
            
    def many_teams(self):
        teams=[]
        C=comb(self.types,self.nagents)
        print("Combinations: "+str(C))
        if C<100:
            for t in combinations(range(self.types),self.nagents):
                teams.append(list(t))
        else:
            for i in range(50):
                teams.append(self.sample())

        return teams
    
    def sample(self):
        n,k=self.nagents,self.types
        return np.sort(np.random.choice(k,n,replace=False))


def test_net():
    a=Net()
    b=Net()
    x=np.array([[1,2,3,4,5,6,7,8]])
    y=np.array([[0]])
    print(a.feed(x))
    print(a.train(x,y))
    print(b.feed(x))
    print(b.train(x,y))

if __name__=="__main__":
    test_net()
    a=all_teams(5)
    print(a)
    
    