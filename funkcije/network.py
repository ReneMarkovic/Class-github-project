"""
Generatorji različnih tipov mrež
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

vse_mreze=['random', 'regular', 'small_world', 'scale_free', 'directed_scale_free']

vse_pozicije=['square','spring','circle','fruchterman','spectral','spiral','hierarchy']

def network(tip, pos, N, #vse - obvezno
            p=0.05, #SW
            k=5, #SW in reg
            d=2, #rand
            m=4, #SF
            alpha=0.41, #DSF
            beta=0.54, #DSF
            gamma=0.05, #DSF
            delta_in=0.2, #DSF
            delta_out=0.2): #DSF

    """Funkcija generira mreže poljubnega tipa. 

    Parametri
    ---------
    tip : string
        Izbrana mreža.
        
    pos : string
        Izbrana postavitev vozlov.
        
    N : int
        Število vseh vozlov.
        
    p : float, optional (default=0.05)
        Verjetnost za daljnosežno povezavo v Small World mreži.
        
    k : int, optional (default=5)
        Število najbližjih sosedov, s katerimi je vozel povezan v 
        Small World in regular mreži.
        
    d : int, optional (default=2)
        Število povezav (degree) vsakega vozla v random mreži.
        
    m : int, optional
        Število vozlov, ki jih ima vsak vozel na zacetku pri skalno 
        neodvisni (scale-free) mreži.
        (preferira vozle z veliko povezavami - prefer high deree)
    
    Parametri usmerjene skalno neodvisne mreže 
    (alpha+beta+gamma=1.0 <--mora vedno veljati)
    alpha : float, optional (default=0.41)
        Verjetnost, da ustvari povezan vozel glede na in-degree 
        distribution v usmerjeni skalno neodvisni mreži.
        
    beta : float, optional (default=0.54)
        Verjetnost, da ustvari povezavo med dvema obstoječima vozloma 
        (eden odvisen od in-degree dist, drugi pa od out-degree dist).
        
    gamma : float, optional (default=0.05)
        Verjetnost, da ustvari povezan vozel glede na out-degree dist.
        
    delta_in : float, optional (default=0.2)
        Obteženost za izbiro vozlov glede na in degree dist.
    
    delta_out : float (default=0.2)
        Obteženost za izbiro vozla glede na out-degree dist.
        
    directed : bool, optional (default=False)
        Če je True, vrne usmerjen graf.
    """
    
    mreza=tip
    
    pos=pos
    
    if mreza=='regular' or mreza=='reg': #regular mreža, povezana samo s k številom najbližjih sosedov
        #k - najblizji sosedi, s katerimi ima povezavo
        v=0.0 #za regular je vedno 0.0, če je več je to Small world mreža (SW spodaj)
        G=nx.watts_strogatz_graph(N,k,v) 
        
    if mreza=='small_world' or mreza=='SW' or mreza=='sw':#small world mreža
        #k - najblizji sosedi, s katerimi je povezan
        #p - verjetnost za daljnosezno povezavo v SW mrezi
        G=nx.watts_strogatz_graph(N,k,p)
    
    if mreza=='random' or mreza=='rand':#random mreza
        #d - st povezav vsakega vozla
        G=nx.random_regular_graph(d, N, seed=None)#random 
        
    if mreza=='scale_free' or mreza=='SF' or mreza=='sf':#skalno neodvisna mreža
        #m - st vozlov, ki jih ima vsak vozel na zacetku (prefer high deree)
        seed=100 #random seed
        G=nx.barabasi_albert_graph(N, m, seed)
    
    if mreza=='directed_scale_free' or mreza=='DSF' or mreza=='dsf':#usmerjena skalno neodvisna mreža  (alpha+beta+gamma=1 <--mora vedno veljati)
        #alpha=0.41 # verjetnost, da ustvari povezan vozel glede na in-degree dist
        #beta=0.54 # verjetnost, da ustvari povezavo med dvema obstoječima vozloma
        #gamma=0.05 #verjetnost, da ustvari povezan vozel glede na out-degree dist
        #delta_in=0.2 #obteženost za izbiro vozlov glede na in degree dist
        #delta_out=0 # obteženost za izbiro vozlov glede na out degree dist
        G=nx.scale_free_graph(N, alpha, beta, gamma, delta_in, delta_out, create_using=None, seed=None)
    
    #lega vozlov
    if pos=='square':
        pos=np.loadtxt('node_coordinates.dat') #za mrezo s 1000 vozli
        
    if pos=='spring':
        pos=nx.spring_layout(G)
        
    if pos=='circular' or pos=='circle':
        pos=nx.circular_layout(G) #krozno
        
    if pos=='fruchterman':
        pos=nx.fruchterman_reingold_layout(G)
        
    if pos=='spectral':
        pos=nx.spectral_layout(G)
        
    if pos=='spiral':
        pos=nx.spiral_layout(G)
        
    if pos=='hierarchy' or pos=='hier':
        degrees = [G.degree[a] for a in G.nodes] #preveri kolikokrat se degree pojavi
        degrees_unique = sorted(list(set(degrees)))#razporedi v razrede
        y_positions = {degrees_unique[i] : i for i in range(len(degrees_unique))} #razporedi v razrede glede na y os
        x_positions = {}
        
        for degree in degrees_unique:
            x_positions[degree] = [a for a in degrees.count(degree) / 2. - np.arange(degrees.count(degree))] #razporedi v posameznem razredu
        
        pos= {}
        
        for node in G.nodes:
            deg = G.degree[node]
            pos[node] = (x_positions[deg].pop(), y_positions[deg])#'appendaj' pozicije (.pop)
    
    atribut=[-1,0,1]
    for i in range(N):
        G.add_node(i,atribut=random.choice(atribut),x=pos[i][0],y=pos[i][1])
     
    nx.write_gexf(G,tip+"_network.gexf")    
    
    return G, pos