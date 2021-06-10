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


'''tmaxit=3000
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

class SEIRSNetworkModel2(SEIRSNetworkModel):
    def run_iteration(self,dt):
        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        if(propensities.sum() > 0):

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate alpha
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            propensities_flat   = propensities.ravel(order='F')
            cumsum              = propensities_flat.cumsum()
            alpha               = propensities_flat.sum()
    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the time until the next event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tau = dt
            self.t += tau
            self.timer_state += tau
    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute which event takes place
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
            transitionNode  = transitionIdx % self.numNodes
            transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]
    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform updates triggered by rate propensities:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
            self.X[transitionNode] = self.transitions[transitionType]['newState']
    
            self.testedInCurrentState[transitionNode] = False
    
            self.timer_state[transitionNode] = 0.0
    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
            # Save information about infection events when they occur:
            if(transitionType == 'StoE'):
                transitionNode_GNbrs  = list(self.G[transitionNode].keys())
                transitionNode_GQNbrs = list(self.G_Q[transitionNode].keys())
                self.infectionsLog.append({ 't':                            self.t,
                                            'infected_node':                transitionNode,
                                            'infection_type':               transitionType,
                                            'infected_node_degree':         self.degree[transitionNode],
                                            'local_contact_nodes':          transitionNode_GNbrs,
                                            'local_contact_node_states':    self.X[transitionNode_GNbrs].flatten(),
                                            'isolation_contact_nodes':      transitionNode_GQNbrs,
                                            'isolation_contact_node_states':self.X[transitionNode_GQNbrs].flatten() })
    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
            if(transitionType in ['EtoQE', 'ItoQI']):
                self.set_positive(node=transitionNode, positive=True)
    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        else:
            tau=dt
            self.t += tau
            self.timer_state += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.tidx += 1
        
        self.tseries[self.tidx]     = self.t
        self.numS[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]        = numpy.clip(numpy.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.numQ_E[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_E), a_min=0, a_max=self.numNodes)
        self.numQ_I[self.tidx]      = numpy.clip(numpy.count_nonzero(self.X==self.Q_I), a_min=0, a_max=self.numNodes)
        self.numTested[self.tidx]   = numpy.clip(numpy.count_nonzero(self.tested), a_min=0, a_max=self.numNodes)
        self.numPositive[self.tidx] = numpy.clip(numpy.count_nonzero(self.positive), a_min=0, a_max=self.numNodes)
        
        self.N[self.tidx]           = numpy.clip((self.numNodes - self.numF[self.tidx]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update testing and isolation statuses
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        isolatedNodes = numpy.argwhere((self.X==self.Q_E)|(self.X==self.Q_I))[:,0].flatten()
        self.timer_isolation[isolatedNodes] = self.timer_isolation[isolatedNodes] + tau

        nodesExitingIsolation = numpy.argwhere(self.timer_isolation >= self.isolationTime)
        for isoNode in nodesExitingIsolation:
            self.set_isolation(node=isoNode, isolate=False)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store system states
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numR'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numF'][self.tidx]        = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.F)
                self.nodeGroupData[groupName]['numQ_E'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_E)
                self.nodeGroupData[groupName]['numQ_I'][self.tidx]      = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Q_I)
                self.nodeGroupData[groupName]['N'][self.tidx]           = numpy.clip((self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numQ_E'][0] + self.nodeGroupData[groupName]['numQ_I'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)
                self.nodeGroupData[groupName]['numTested'][self.tidx]   = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.tested)
                self.nodeGroupData[groupName]['numPositive'][self.tidx] = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.positive)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infections is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(self.t >= self.tmax or (self.total_num_infected(self.tidx) < 1 and self.total_num_isolated(self.tidx) < 1)):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True





def SEIR(G,ponovitev,tmaxit,beta,sigma,gamma,initE,N,tip,dt):
    Nlist=[N for i in range(tmaxit+1)]
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
        
        model = SEIRSNetworkModel2(G, beta=beta, sigma=sigma, gamma=gamma, initE=initE,store_Xseries = True)   
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
            model.run_iteration(dt)
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
        #print(len(Nlist),len(list0),len(list1),len(list2),len(list3),len(list4),len(list5))
        for i in range(tmaxit+1):
            difference = Nlist[i]-list2[i]-list3[i]-list4[i]-list6[i]
            Rlist.append(difference)
        #print(list1)
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
        plt.show()
        plt.plot(list0,list1)
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