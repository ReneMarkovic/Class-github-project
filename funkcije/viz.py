import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.animation import FuncAnimation,PillowWriter
import network
import analysis

d,df=analysis.full_analysis()

'''#Izris mreže
def mreža(datoteka_index,tip,pos,N):
    G, pos=network.network(tip,pos,N)
    
    df=pd.read_csv(datoteka_index)
    
    A=df[df["Value"] == "[1]"] #vozli, ki imajo stanje [1]
    
    B=df[df["Value"] == "[2]"] #vozli, ki imajo stanje [2]
    
    co=["red","blue"]
    col=[]
    
    nodesize=[]   #velikost vozlov določimo v skladu s tem koliko povezav imajo
    for i in range(len(pos)):
        nodesize.append(5*(G.degree(i)+1))
        
    for node in G:
        if node in B['index'].values:
            col.append(co[0])
        else:
            col.append(co[1])    
    
    MREŽA=nx.draw(G,pos,node_size=nodesize,node_color=col,alpha=0.9,linewidths=0.2,edge_color='black',width=1)
    plt.axis('off') 
    plt.savefig("MREZA_SW.jpg",dpi=100,bbox_inches='tight')
    plt.show()
    plt.close()
    
    return MREŽA

mreža(r'SEIR_results/Tip=SW,beta=0.45,sigma=0.20,gamma=0.14,InitE=100.00/pon=_0_Nodes.csv','small_world','fruchterman',1000)
#datoteka_index=SEIR_results/Tip=SW,beta=0.45,sigma=0.20,gamma=0.14,InitE=100.00/pon=_0_Nodes.csv'''


#Časovni potek epidemije
def časovni_potek(datoteka):
    plt.style.use('seaborn-bright')
    df=pd.read_csv(datoteka)
    df.plot(x="Time",y=["S","E","I","R"])
    plt.xlabel("čas")
    plt.ylabel("število")
    plt.savefig("ČASOVNI_POTEK.jpg",dpi=100,bbox_inches='tight')
    plt.show()
    plt.close()
    
    return časovni_potek
časovni_potek('SEIR_results/Tip=SW,beta=0.45,sigma=0.20,gamma=0.14,InitE=100.00/pon=_0.csv')


#Animacija časovnega poteka
def animacija(datoteka):
    plt.style.use('seaborn-bright')
    fig, ax = plt.subplots(figsize=(15,10)) 
    
    AAPL_STOCK = pd.read_csv(datoteka)
    
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
        ax.set_xlim(0,AAPL_STOCK.iloc[-1,1])  
        ax.set_ylim(0,AAPL_STOCK.iloc[0,3])
    
    
    def animation(i):
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
    
    animation = FuncAnimation(fig, func=animation,init_func=init,frames=len(AAPL_STOCK.index))
    writer = PillowWriter(fps=25)  
    animation.save("ANIMACIJA.gif", writer=writer)
    plt.show()
    plt.close()
    
    return animation
animacija('SEIR_results/Tip=SW,beta=0.45,sigma=0.20,gamma=0.14,InitE=100.00/pon=_0.csv')


#Porazdelitev števila povezav
def porazdelitev():
    a=d["0"]["Network"]["degree_dist"][0]
    b=d["0"]["Network"]["degree_dist"][1]
    plt.scatter(a,b)
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.savefig("PORAZDELITEV.jpg",dpi=100,bbox_inches='tight')
    plt.show()
    plt.close()
    
    return porazdelitev
porazdelitev()


#Agregacija podatkov
def agreg():
    meanS=d["0"]["avg_SEIR"]["S"][0]
    meanE=d["0"]["avg_SEIR"]["E"][0]
    meanI=d["0"]["avg_SEIR"]["I"][0]
    meanR=d["0"]["avg_SEIR"]["R"][0]
    num_rows = np.shape(meanS)[0]
    plt.figure()
        
    plt.plot(np.arange(num_rows),meanS,label='S')
    plt.plot(np.arange(num_rows),meanE,label='E')
    plt.plot(np.arange(num_rows),meanI,label='I')
    plt.plot(np.arange(num_rows),meanR,label='R')
    stdS=d["0"]["avg_SEIR"]["S"][1]
    stdE=d["0"]["avg_SEIR"]["E"][1]
    stdI=d["0"]["avg_SEIR"]["I"][1]
    stdR=d["0"]["avg_SEIR"]["R"][1]
    y1S=meanS-2*stdS
    y2S=meanS+2*stdS
    y1E=meanE-2*stdE
    y2E=meanE+2*stdE
    y1I=meanI-2*stdI
    y2I=meanI+2*stdI
    y1R=meanR-2*stdR
    y2R=meanR+2*stdR
    plt.fill_between(np.arange(num_rows),y1S,y2S,alpha=0.2)
    plt.fill_between(np.arange(num_rows),y1E,y2E,alpha=0.2)
    plt.fill_between(np.arange(num_rows),y1I,y2I,alpha=0.2)
    plt.fill_between(np.arange(num_rows),y1R,y2R,alpha=0.2)
    plt.legend()
    plt.savefig("AGREGACIJA.jpg",dpi=100,bbox_inches='tight')
    plt.show()
    plt.close()
    
    return agreg
agreg()
