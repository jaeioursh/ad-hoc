
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

def plot_graph(graph):
    graph-=np.min(graph)
    graph/=np.max(graph)/5

    I,J,K=graph.shape
    Es=[]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if i!=j:
                    Es.append(graph[i,j,k])
    A=np.ones((len(graph),len(graph)),dtype=int)*graph.shape[-1]
    print(A)
    d=np.eye(len(graph))
    A[d>0]=0
    G=nx.from_numpy_matrix(A,True,nx.MultiDiGraph)
    pos = nx.circular_layout(G)
    '''
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=20,
        width=2,
    )
    '''
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    ax = plt.gca()
    
    for e,ed in zip(G.edges,Es):
        
        print(e)
        ax.annotate("",
            
            xy=pos[e[0]], xycoords='data',
            xytext=pos[e[1]], textcoords='data',
            arrowprops=dict(arrowstyle="->", color="0.1",
                            shrinkA=5, shrinkB=5,
                            patchA=None, patchB=None,
                            lw=int(ed),
                            
                            connectionstyle="arc3,rad=rrr".replace('rrr',str(0.1*(e[2]+1))
                            ),
                            ),
            )
    #pc = mpl.collections.PatchCollection(edges)
    #plt.colorbar(pc)
    plt.show()
N=4 
couple=2


poi=np.array([1,2,3,4,0])

def G(act):
    count=np.bincount(act)
    count=np.pad(count,(0,len(poi)-len(count)),constant_values=(0,0))
    return np.sum(poi[count>=couple])
def D1(act):
    act=np.array(act)
    gg=G(act)
    d1=np.zeros(len(act))
    for i in range(len(act)):
            ACT=act.copy()
            ACT[i]=len(poi)-1
            g=G(ACT)
            d1[i]=gg-g 

    return d1

def D2(act):
    act=np.array(act)
    gg=G(act)
    d2=np.zeros((len(act),len(act)))
    for i in range(len(act)):
        for j in range(i):
            ACT=act.copy()
            ACT[i]=len(poi)-1
            ACT[j]=len(poi)-1
            g=G(ACT)
            d2[i,j]=gg-g 
            d2[j,i]=gg-g 
    return d2
lr=0.0001
graph=np.zeros((N,N,len(poi)-1))

gs=[]

for i in range(100):
    if 0:
        act=np.random.randint(0,len(poi)-1,N)
    else:
        ACT=np.sum(graph,axis=1)
        act=[]
        for a in ACT:
            a=a.copy()
            a-=np.min(a)-.1
            a/=np.sum(a)
            if i%2==1:
                #choice=np.random.choice(np.arange(0,len(a)),1)[0]
                choice=np.random.choice(np.arange(0,len(a)),1,p=a)[0]
            else:
                choice=np.argmax(a)
            act.append(choice)
        
    #print("a: ",act)
    g=G(act)
    if i%2==0:
        gs.append(g)
    else:
        #d1=D1(act)
        d2=D2(act)
        for i in range(len(act)):
            for j in range(len(act)):
                if j!=i:
                    graph[i,j,act[i]]*=1-lr
                    graph[i,j,act[i]]+=lr*d2[i,j]

    #print(act,g)
    #print(d)
    #print()
#gs=np.convolve(gs,np.ones(25)/25,mode="valid")
plot_graph(graph)
plt.scatter(np.arange(0,len(gs)),gs)
plt.show()