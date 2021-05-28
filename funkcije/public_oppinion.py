import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

#Metropolis algoritem

'''n = 100
m = 3

G = nx.barabasi_albert_graph(n, m, seed = None)'''

def E(G,op):
    E=np.zeros(N,float)
    Eavg=0.0
    for i in G:
        if G.degree(i)>0:
            xx=G.nodes[i]['atribut']
            nn=[n for n in G.neighbors(i)]
            #nn=len(G.neighbors(i))
            for k in G.neighbors(i):
                E[i]+=abs(G.nodes[k]['atribut']-xx)
            E[i]=E[i]/(float(len(nn)))
            Eavg+=E[i]
    return Eavg

def opinion(G,za,nevem,proti,rep,flesibilnost):
    N=len(G)
    op=np.zeros(N,float)
    
    #Ali se vozlišča želijo cepit, se ne želijo cepit oz. so neopredeljena na začetku simulacije
    for i in G:
        if random.random()<za:
            G.nodes[i]['atribut']=1
        elif random.random()<za+proti:
            G.nodes[i]['atribut']=-1
        else:
            G.nodes[i]['atribut']=0
          
    Eold=E(G)
    print(Eold)
    for t in range(N*rep):
        iz=random.randint(0,len(G)-1)
        temp=G.nodes[iz]['atribut']
        G.nodes[iz]['atribut']=random.choice([-1,0,1])
        Enew=E(G)
        if(Enew>Eold):
            G.nodes[iz]['atribut']=temp
        else:
            if (Metropolis,flesibilnost)
            
            else:
                G.nodes[iz]['atribut']=temp
   return G
