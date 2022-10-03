import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
compact=1
if compact:
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -14
from teaming import logger
data=[]
err=[]
AGENTS=5
ROBOTS=4


i=12

fname="tests/vary/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-3.pkl"
log = logger.logger()

#log.load("tests/evo38-5.pkl")
log.load(fname)

r=len(log.pull("reward"))

times=np.array(log.pull("ctime"))
times=times.flatten()
print(times.shape)
times=times.reshape((r,-1))
probs=np.array([np.bincount(t,minlength=30).astype(float)/r for t in times])
plt.imshow(probs.T,interpolation="none",aspect='auto')
plt.xlabel("Generation")
plt.ylabel("Time Step")
plt.colorbar(label="Probability of Being Chosen")
plt.tight_layout()
plt.show()
