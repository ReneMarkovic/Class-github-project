import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random


n = 100
m = 3

G = nx.barabasi_albert_graph(n, m, seed = None)
#nx.draw(G)
#plt.show()


def E(G,op):
    E=np.zeros(N,float)
    print(E)
    Eavg=0.0
    for i in G:
        xx=op[i]
        if G.degree(i)>0:
            nn=[n for n in G.neighbors(i)]
            #nn=len(G.neighbors(i))
            for k in G.neighbors(i):
                E[i]+=abs(op[k]-xx)
            E[i]=E[i]/(float(len(nn)))
            Eavg+=E[i]
    return Eavg

def opinion(G,op):
    Eold=E(G,op)
    iz=random.randint(0,len(G)-1)
    temp=op[iz]
    op[iz]=abs(1-op[iz])
    Enew=E(G,op)
    if(Enew>Eold):
        op[iz]=temp
N=500
delta=1.7
theta=5
k_avg=5.0

DC = 0.9 #delez cepljenih
ii=0
c=["gray","orange","red","green"]
x=np.zeros(N,float)
y=np.zeros(N,float)
for i in range(N):
    x[i]=random.uniform(0,1)
    y[i]=random.uniform(0,1)

while delta<1.8:
    op=np.zeros(N,float)
    G=nx.Graph()
    #MORITA_MODEL(G,x,y,theta,delta,k_avg)
    skupine=nx.find_cliques(G)
    pripadnost={}
    for i in skupine:
        #print(i[0],len(i))
        if len(i)>1:
            if i[0] in pripadnost:
                for ii in i[1::]:
                    if ii not in pripadnost[i[0]]:
                        pripadnost[i[0]].append(ii)
            else:
                pripadnost[i[0]]=[]
                for ii in i[1::]:
                    if ii not in pripadnost[i[0]]:
                        pripadnost[i[0]].append(ii)
    iz=0
    max_skupina=0
    for i in pripadnost:
        if len(pripadnost[i])>max_skupina:
            max_skupina=len(pripadnost[i])
            iz=pripadnost[i]
    
    pos=nx.get_node_attributes(G,'pos')
    ss=[]
    for i in G:
        ss.append(G.degree(i)*3)
    col=[]
    for i in G:
        if i in iz:
            op[i]=0
            col.append(c[0])
        else:
            op[i]=1
            col.append(c[1])
    print(E(G,op))
    '''for i in op:
        if random.random()>0.9:
            col.append(c[0])
        else:
            col.append(c[1])'''
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    nx.draw(G,pos,node_size=ss,node_color=col)
    plt.title("delta = %.1f"%(delta))
    
    ##Evolving the opinion
    for t in range(100):
        opinion(G,op)
    col=[]
    for i in G:
        if op[i]==0:
            col.append(c[0])
        else:
            col.append(c[1])
    plt.subplot(122)
    nx.draw(G,pos,node_size=ss,node_color=col)
    plt.title("delta = %.1f"%(delta))
    plt.show()
    ii+=1
    delta+=0.1