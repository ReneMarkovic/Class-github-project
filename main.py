import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from funkcije import network
from funkcije import public_oppinion as po

tip='SW'
oblika='circle'
N=100

G, pos=network.network(tip,oblika, N)

##Evolucija javnega mnenja
#init parametri za model javnega mnenja#
za=0.1
proti=0.1
nevem=1-za-proti
G=po.opnion(G,za,nevem,proti)
#Rezultat je mre

#SEIR - #
