import numpy as np
import pyximport
pyximport.install()
from teaming import logger
from teaming.learnmtl import Net
AGENTS=4
ROBOTS=3

i=42

q=4
fname="tests/vary/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl"

log = logger.logger()
#log.load("tests/evo38-5.pkl")
log.load(fname)
hist=log.pull("hist")[0]
print(len(hist[0]))

net=Net()


S,A,D=[],[],[]
            
for samp in hist[0]:
    S.append(samp[0])
    A.append(samp[1])
    D.append([samp[2]])

S,A,D=np.array(S),np.array(A),np.array(D)
Z=np.hstack((S,A))
net.train(Z,D,10,2)
d=net.feed(Z)
print(np.mean(np.square(d-D)))
print(np.count_nonzero(D))
print(np.mean(D),np.mean(d))
print(np.min(D),np.min(d))
print(np.max(D),np.max(d))
print(d-D)