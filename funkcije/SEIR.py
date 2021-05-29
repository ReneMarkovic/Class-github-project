import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import EoN
from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import pandas as pd

G=nx.watts_strogatz_graph(n=1000, k=4, p=0.6)
'''#plt.figure(figsize=(40,40))
#nx.draw_circular(g, with_labels = True)

E = g.number_of_edges()
#initializing random weights
w = [random.random() for i in range(E)]
s = max(w)
w = [ i/s for i in w ] #normalizing
len(w)
k = 0
for i, j in g.edges():
    g[i][j]['weight'] = w[k]
    k+=1
import matplotlib.pyplot as plt
edgewidth = [d['weight'] for (u,v,d) in g.edges(data=True)]
# layout
#pos = nx.spring_layout(G, iterations=50)
pos = nx.spring_layout(g)
labels = {}
for i in range(100):
    labels[i] = i
# rendering
plt.figure(figsize=(40,40))
nx.draw_networkx_nodes(g, pos)
nx.draw_networkx_edges(g, pos, width=edgewidth, node_size=500)
nx.draw_networkx_labels(g, pos, labels)
plt.axis('off')

# gamma rate of recovery
# beta transmision
# sigma rate progression of infection


gamma = 0.2  # rate of recovery
beta = 1.2 #transmision

r_0 = beta/gamma
print(r_0)
N = 100 # population size
I0 = 1   # intial n° of infected individuals 
R0 = 0
S0 = N - I0 -R0
pos = nx.spring_layout(g)
nx_kwargs = {"pos": pos, "alpha": 0.7} #optional arguments to be passed on to the
#networkx plotting command.
print("doing Gillespie simulation")
#SIR
sim = EoN.Gillespie_SIR(g, tau = beta, gamma=gamma, rho = I0/N, transmission_weight="weight", return_full_data=True)
print("done with simulation, now plotting")'''
#SEIR
#sigma = 1 # rate progression of infection
#input = np.loadtxt("network_data\\SF_net_matrix_N1000.dat")




model = SEIRSNetworkModel(G, beta=0.45, sigma=1/5.1, gamma=1/7, initE=100,store_Xseries = True)

#model.run(T = 1000,print_interval=1.00)

tmaxit=2000
list0=np.arange(0,tmaxit+1,1)
for t in range (tmaxit):    
    model.run_iteration()
    if(t%100==0):
        print(t)
        pass
#print(list0)
list00=list0.tolist()
#model.X
#print(model.X)
t = model.tseries
model.figure_basic(plot_R='line')
S = model.numS            # time series of S counts
E = model.numE            # time series of E counts
I = model.numI            # time series of I counts
R = model.numR            # time series of R counts
F = model.numF            # time series of F counts
Q_E = model.numQ_E        # time series of Q_E counts
Q_I = model.numQ_I        # time series of Q_I counts
#print(list0)
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
#print(list0,list1)
#print(len(list0),len(list1),len(list2),len(list3),len(list4),len(list5))
#plt.plot(list1)
'''plt.plot(list2)
plt.show()
plt.plot(list3)
plt.show()
plt.plot(list4)
plt.show()
plt.plot(list5)
plt.show()'''
#print(len(list0),len(list2))

df={
    "Time":list1,
    "S":list2,
    "E":list3,
    "I":list4,
    "R":list5
}

df2={
    "Time":list00,
    "S":list2,
    "E":list3,
    "I":list4,
    "R":list5
}

df=pd.DataFrame.from_dict(df)
df = df.set_index('Time')
df.to_csv("SEIR_results.csv")

df2=pd.DataFrame.from_dict(df2)
df2 = df2.set_index("Time")
df2.to_csv("SEIR_results2.csv")

f=open('outputt.txt','w')
for element in list1:
    f.write(str(element) + "\n")
f.close()
f=open('outputS.txt','w')
for element in list2:
    f.write(str(element) + "\n")
f.close()
f=open('outputE.txt','w')
for element in list3:
    f.write(str(element) + "\n")
f.close()
f=open('outputI.txt','w')
for element in list4:
    f.write(str(element) + "\n")
f.close()
f=open('outputR.txt','w')
for element in list5:
    f.write(str(element) + "\n")
f.close()

#vizualizacija SIR
'''for i in range(0,5,1):
    sim.display(time = i,  **nx_kwargs)
    plt.axis('off') 
    plt.title("Iteration {}".format(i))
    plt.draw()
    
    
    with open('your_file.txt', 'w') as f:
    for item in my_list:
        f.write("%s\n" % item)'''

