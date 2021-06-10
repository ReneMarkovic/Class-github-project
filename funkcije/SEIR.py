import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import pandas as pd
import os

tmaxit=1000'''
beta=0.45
sigma=1/5.1
gamma=1/7
initE=100
ponovitev=1
N=1000 # velikost mreže
k=5 # povprečno povezanost
p=0.1 # verjetnost za dajnosezno povezavi v SW mrezi
seed=100
G=nx.watts_strogatz_graph(N,k,p)
tip='SW'''
Nlist=[N for i in range(tmaxit+1)]
#print(Nlist)
def SEIR(G,ponovitev,tmaxit,beta,sigma,gamma,initE,N,tip):
    G=G
    '''tmaxit=200
    beta=0.45
    sigma=1/5.1
    gamma=1/7
    initE=100'''
    ponovitev=ponovitev
    tmaxit=tmaxit
    beta=beta
    sigma=sigma
    gamma=gamma
    initE=initE
    potf="SEIR_results\\Tip=%s,beta=%.2f,sigma=%.2f,gamma=%.2f,InitE=%.2f\\"%(tip,beta,sigma,gamma,initE)
    
    for pon in range(ponovitev):
        
        nodeindex=np.arange(0,N,1)
        
        model = SEIRSNetworkModel(G, beta=beta, sigma=sigma, gamma=gamma, initE=initE,store_Xseries = True)   
        X=model.X
        #model.X[0]=5
        for node in range(0,N,1):
            if(model.X[node]==8):
                    model.X[node]=1
            if(model.X[node]==9):
                    model.X[node]=1
        
        try:
            os.mkdir(potf)
            
        except:
            pass
        
        if pon==1:
            f=open(potf+'parameters.dat', 'w')
            f.write("%f %f %f %d %d\n"%(beta,sigma,gamma,initE,N))
            f.close()
            nx.write_gexf(G,potf+tip+"_network.gexf")
        
        listnode=X.tolist()
        dfnode={
            "index":nodeindex,
            "Value":listnode,
        }
        dfnode=pd.DataFrame.from_dict(dfnode)
        dfnode = dfnode.set_index('index')
        dfnode.to_csv(potf+"pon=_%d_Nodes.txt"%(pon))
        
        
        list0=np.arange(0,tmaxit+1,1)
        for t in range (tmaxit):    
            model.run_iteration()
            rr=random.random()
            if(rr<0.01):
                rrnode=random.randint(0,N-1)
                #print(rrnode)
                if(model.X[rrnode]==1):
                    model.X[rrnode]=5
        pass
        #print(list0)
        X2=model.X
        list00=list0.tolist()
        #model.X
        #print(model.X)
        t = model.tseries
        S = model.numS            # time series of S counts
        E = model.numE            # time series of E counts
        I = model.numI            # time series of I counts
        R = model.numR            # time series of R counts
        F = model.numF            # time series of F counts
        Q_E = model.numQ_E        # time series of Q_E counts
        Q_I = model.numQ_I        # time series of Q_I counts
        #print(type(S))str(t),str(S),str(E),
        
        #results=str(t),str(model.numS),str(model.numE),str(model.numI),str(model.numR),str(model.numF),str(model.numQ_E),str(model.numQ_I)
        list1 =t.tolist()
        list2 =S.tolist()
        list3 =E.tolist()
        list4 =I.tolist()
        list5 =R.tolist()
        list6 =F.tolist()
        list7 =Q_E.tolist()
        list8 =Q_I.tolist()
        Rlist=[]
        print(len(Nlist),len(list0),len(list1),len(list2),len(list3),len(list4),len(list5))
        for i in range(tmaxit+1):
            difference = Nlist[i]-list2[i]-list3[i]-list4[i]-list6[i]
            Rlist.append(difference)
        print(Rlist)
        listnodeafter=X2.tolist()

        #print(list0,list1)
        
        #plt.plot(list1)
        '''plt.plot(list2)
        plt.show()
        plt.plot(list3)
        plt.show()
        plt.plot(list4)
        plt.show()
        plt.plot(list6)
        plt.show()
        plt.plot(Rlist)
        plt.show()'''
        #print(len(list0),len(list2))
        #print(listnode)
        #print(listnodeafter)
        
        df={
            "Iter":list00,
            "Time":list1,
            "S":list2,
            "E":list3,
            "I":list4,
            "R":Rlist
        }
        
        df=pd.DataFrame.from_dict(df)
        df = df.set_index('Iter')
        df.to_csv(potf+"pon=_%d.csv"%(pon))
        
        
        
        dfnode2={
            "index":nodeindex,
            "Value":listnodeafter,
        }
        dfnode2=pd.DataFrame.from_dict(dfnode2)
        dfnode2 = dfnode2.set_index('index')
        dfnode2.to_csv(potf+"pon=_%d_NodesAfter.txt"%(pon))
        print(pon)


#SEIR(G,ponovitev,tmaxit,beta,sigma,gamma,initE,N,tip)