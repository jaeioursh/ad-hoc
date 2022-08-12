"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
from os import killpg
import numpy as np
import sys
import multiprocessing as mp


from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain

#import mods
from teaming.learnmtl import learner
from sys import argv
import pickle
#import tensorflow as tf

def rand_loc(n):
    x,y=np.random.random(2)
    pos=[[x,y]]
    while len(pos)<6:
        X,Y=np.random.random(2)
        for x,y in pos:
            dist=((X-x)**2.0+(Y-y)**2.0 )**0.5
            if dist<0.2:
                X=None 
                break
        if not X is None: 
            pos.append([X,Y])
    
    return np.array(pos)


#print(vals)
def make_env(nagents):
    vals =np.array([0.8,1.0,0.6,0.3,0.2,0.1])
    
    
    pos=np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.5],
        [0.0, 0.5],
        [1.0, 0.0]
    ])

    #pos=rand_loc(6)#np.random.random((6,2))
    #vals=np.random.random(6)/2.0
    print(vals)

    sim = RoverDomainGym(nagents,30,pos,vals)
 


    sim.data["Coupling"]=2
    sim.data['Number of Agents']=nagents

    obs=sim.reset()
    return sim


import time

def test1(trial,k,n,train_flag):
    #print(np.random.get_state())[1]    
    np.random.seed(int(time.time()*100000)%100000)
    env=make_env(n)
 
    OBS=env.reset()

    controller = learner(n,k,env)
    

    for i in range(4001):

        controller.randomize()
        
        if i%50==0:
            controller.test(env)

        r=controller.run(env,train_flag)# i%100 == -10)
        print(i,r,controller.team[0])
        
            
        if i%50==0:
            #controller.save("tests/q"+str(frq)+"-"+str(trial)+".pkl")
            #controller.save("logs/"+str(trial)+"r"+str(16)+".pkl")
            #controller.save("tests/jj"+str(121)+"-"+str(trial)+".pkl")
            #controller.log.clear("hist")
            #controller.put("hist",controller.hist)
            controller.save("tests/vary/"+str(k)+"-"+str(n)+"-"+str(trial)+"-"+str(train_flag)+".pkl")



if 0:
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...
    test1(42,5,4,1)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    
else:
    for train in [1]:
        procs=[]
        k=5
        n=4
        for i in range(8):
            p=mp.Process(target=test1,args=(i,k,n,train))
            p.start()
            time.sleep(0.05)
            procs.append(p)
            #p.join()
        for p in procs:
            p.join()
