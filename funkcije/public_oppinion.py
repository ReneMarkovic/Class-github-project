import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import math

#Metropolis algoritem
def Metropolis(Enew, Eold, fleksibilnost):
    randomNumber = random.uniform(0, 1)
    if (randomNumber < math.exp((Enew - Eold) / fleksibilnost)):
        return True
    else:
        return False
'''n = 100
m = 3

G = nx.barabasi_albert_graph(n, m, seed = None)'''

def E(G,op, N):
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

def opinion(G,za,nevem,proti,rep,fleksibilnost, N):
    N = len(G)
    op=np.zeros(N,float)
    
    #Ali se vozlišča želijo cepit, se ne želijo cepit oz. so neopredeljena na začetku simulacije
    for i in G:
        if random.random()<za:
            G.nodes[i]['atribut']=5
        elif random.random()<za+proti:
            G.nodes[i]['atribut']=8
        else:
            G.nodes[i]['atribut']=9
          
    Eold=E(G,op,N)
    #print(Eold)
    for t in range(N*rep):
        iz=random.randint(0,len(G)-1)
        temp=G.nodes[iz]['atribut']
        G.nodes[iz]['atribut']=random.choice([5,8,9])
        Enew=E(G,op,N)
        
        if(Enew < Eold):
            Eold=Enew
        else:
            if(Metropolis(Enew, Eold, fleksibilnost)):
                Eold=Enew
            else:
                G.nodes[iz]['atribut'] = temp
    return G
