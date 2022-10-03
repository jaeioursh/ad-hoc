#alignment/vis
from mtl import make_env

import numpy as np
import matplotlib.pyplot as plt
import torch

from teaming import logger
from teaming.learnmtl import Net

q=3
i=4
AGENTS=5
ROBOTS=4

fname="tests/very/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl"

log = logger.logger()
log.load(fname)
env=make_env(ROBOTS)
pos=log.pull("position")
INDEX=0
TEAM=2

#env.reset()
env.data["Agent Positions"]=pos[-1][TEAM]
env.data["Agent Position History"]=np.array([env.data["Agent Positions"]])
env.data["Steps"]=-1
env.data["Observation Function"](env.data)
#print(env.data["Agent Observations"])

print(env.data["Agent Positions"])


env.data["Reward Function"](env.data)
print(env.data["Global Reward"])
print(env.data["Agent Rewards"])
#print(env.data["Agent Orientations"])

net=Net()
net.model.load_state_dict(torch.load(fname+".mdl")[INDEX])