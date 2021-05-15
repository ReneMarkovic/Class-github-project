import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



N=1000 #velikost mreže

#napiši mrežo, katero želiš izvoziti (rand - random, SW - Small World, SF - Scale free, DSF - directed scale free)
mreza="DSF"


if mreza=="rand":#random mreza
    d=2 #st povezav vsakega vozla

    G=nx.random_regular_graph(d, N, seed=None)#random

if mreza=="SW":#small world mreža
    k=5 #najblizji sosedi
    p=0.2 #verjetnost za daljnosezno povezavo v SW mrezi
    
    G=nx.watts_strogatz_graph(N,k,p) 

if mreza=="SF":#skalno neodvisna mreža
    
    m=4 #st vozlov, ki jih ima vsak vozel na zacetku (prefer high deree)
    seed=100 #random seed
    
    G=nx.barabasi_albert_graph(N, m, seed)

if mreza=="DSF":#usmerjena skalno neodvisna mreža  (alpha+beta+gamma=1 <--mora vedno veljati)
    alpha=0.41 # verjetnost, da ustvari povezan vozel glede na in-degree dist
    beta=0.54 # verjetnost, da ustvari povezavo med dvema obstoječima vozloma
    gamma=0.05 #verjetnost, da ustvari povezan vozel glede na in-degree dist
    delta_in=0.2 #obteženost za izbiro vozlov glede na in degree dist
    delta_out=0 # obteženost za izbiro vozlov glede na out degree dist
    
    G=nx.scale_free_graph(N, alpha, beta, gamma, delta_in, delta_out, create_using=None, seed=None)

#lega vozlov
#pos=np.loadtxt("node_coordinates.dat") #za mrezo s 1000 vozli
#pos=nx.circular_layout(G) #krozno
pos=nx.spring_layout(G)


#izris grafa
nx.draw_networkx(G,pos,node_size=20,with_labels=False)
plt.savefig("network_data\\%s_network_N%d.jpg"%(mreza,N),dpi=200,bbox_inches='tight')
plt.show()
plt.close()


#izpis vseh parov v mrezi (povezani - 1, nepovezani - 0)
out=open("network_data\\%s_net_matrix_N%d.dat"%(mreza,N),"w+")
for i in range(len(G)): #zapise vse pare vozlov, ki so med sabo povezani
    for j in range(len(G)):
        if (G.has_edge(i,j)):#ce obstaja povezava med itim in jtim vozlom
            print(i,j,1,file=out)
        else:
            print(i,j,0,file=out)

out.close()

#izpis povezanih parov
out=open("network_data\\%s_net_connected_N%d.dat"%(mreza,N),"w+")
for i in range(len(G)): #zapise vse pare vozlov, ki so med sabo povezani
    for j in range(len(G)):
        if (G.has_edge(i,j)):#ce obstaja povezava med itim in jtim vozlom
            print(i,j,file=out)

out.close()

