import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.animation import FuncAnimation,PillowWriter
#import community
import analysis

#Izris mreže
df=pd.read_csv('SEIR_results/Tip=SW,beta=0.45,sigma=0.20,gamma=0.14,InitE=100.00/pon=_0_Nodes.csv')
#print(df)

A=df[df["Value"] == "[1]"] #vozli, ki imajo stanje [1]
#print(A)

B=df[df["Value"] == "[2]"] #vozli, ki imajo stanje [2]
#print(B)

co=["red","blue"]
col=[]

CMxyz=np.loadtxt("network_data/SW_net_matrix_N1000.dat") # uvoz matrike
N=int(np.max(CMxyz[:,0]))+1  # razbere število vozlov, maksimalna vrednost v prvem stolpcu, dodamo 1, ker začnemo štet z 0
print ('Stevilo vozlov:',N)

CM=np.zeros([N,N],int) # ustvari prazno 2D matriko
for c in CMxyz:        # sprehajamo se po vrsticah
    CM[int(c[0])][int(c[1])]=int(c[2])   # pretvorba xyz -> 2D matrika
    
    
G=nx.Graph()   # ustvari prazno mrežo G
for i in range(N):
    G.add_node(i)        # ustvari vozel
for i in range(N):       # zanki čez celo CM matriko
    for j in range(N):
        if (CM[i,j]>0):
            G.add_edge(i,j)  # ustvari med i&j povezavo, če tako "pravi" CM
            
pos=nx.fruchterman_reingold_layout(G,k=1.0/np.sqrt(N))
#pos=nx.spring_layout(G)
#pos=nx.circular_layout(G)
#pos=nx.spectral_layout(G)
#pos=nx.spiral_layout(G)

nodesize=[]   #velikost vozlov določimo v skladu s tem koliko povezav imajo
for i in range(len(pos)):
    nodesize.append(10*np.sqrt(G.degree(i)+1))
    
for node in G:
    if node in B['index'].values:
        col.append(co[0])
    else:
        col.append(co[1])    

nx.draw(G,pos,node_size=nodesize,node_color=col,alpha=0.9,linewidths=0.2,edge_color='black',width=1)
plt.axis('off') 
plt.savefig("MREZA_SW.jpg",dpi=100,bbox_inches='tight')
plt.show()
plt.close()


#Časovni potek epidemije
df=pd.read_csv('SEIR_results/beta=_0.45,sigma=_0.20,gamma=_0.14,InitE=_100.00/pon=_0.csv')
#print(df)
df.plot(x="Time",y=["S","E","I","R"])
plt.xlabel("čas")
plt.ylabel("število")
plt.savefig("CASOVNI POTEK.jpg",dpi=100,bbox_inches='tight')
plt.show()
plt.close()


#Animacija časovnega poteka
#plt.style.use('seaborn-bright')
fig, ax = plt.subplots() 

x = []
yS = []
yE = []
yI = []
yR = []

ln1, = plt.plot([], [],label='S')  
ln2, = plt.plot([], [],label='E')
ln3, = plt.plot([], [],label='I')  
ln4, = plt.plot([], [],label='R')


def init():  
    ax.set_xlim(0,5)  
    ax.set_ylim(0,1000)


def animation(i):
    AAPL_STOCK = pd.read_csv('SEIR_results/beta=_0.45,sigma=_0.20,gamma=_0.14,InitE=_100.00/pon=_0.csv')
  
    x=(AAPL_STOCK[0:i]['Time'])
    yS=(AAPL_STOCK[0:i]['S'])
    yE=(AAPL_STOCK[0:i]['E'])
    yI=(AAPL_STOCK[0:i]['I'])
    yR=(AAPL_STOCK[0:i]['R'])

    ln1.set_data(x, yS)  
    ln2.set_data(x, yE)
    ln3.set_data(x, yI)  
    ln4.set_data(x, yR)
  
    plt.xlabel("čas")
    plt.ylabel("število")
    plt.legend()

animation = FuncAnimation(fig, func=animation,init_func=init)
writer = PillowWriter(fps=25)  
animation.save("ANIMACIJA.gif", writer=writer)
plt.show()


#Porazdelitev števila povezav
porazdelitev=analysis.network_parameters('path')
plt.scatter(a_,b_)
plt.xlabel("k")
plt.ylabel("P(k)")
plt.savefig("porazdelitev povezav.jpg",dpi=100,bbox_inches='tight')
plt.show()
plt.close()


#Agregacija podatkov
agreg=analysis.average_SEIR(location)
df=pd.read_csv('SEIR_results/beta=_0.45,sigma=_0.20,gamma=_0.14,InitE=_100.00/pon=_0.csv')
plt.figure()
mean=ret_list_I[:,0]
plt.plot(df['Time'].values,mean)
std=ret_list_I[:,1]
y1=mean-std
y2=mean+std
plt.fill_between(df['Time'].values,y1,y2,alpha=0.2)
