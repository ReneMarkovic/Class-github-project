import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from funkcije import network
from funkcije import public_oppinion as po
from funkcije import SEIR as seir

#----------1 DEL---------------#
tip='SW'
oblika='circle'
N=100

G, pos=network.network(tip,oblika, N)



#----------2 DEL---------------#
##Evolucija javnega mnenja
#init parametri za model javnega mnenja#
za=0.1
proti=0.1
nevem=1-za-proti
G=po.opnion(G,za,nevem,proti)
#Rezultat je mreza, kjer so menja vozlišč določena po principu minimalne energije

#----------3 DEL---------------#
seir(G)

#----------3 DEL---------------#
