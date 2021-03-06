import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from funkcije import network
from funkcije import public_oppinion as po
from funkcije import SEIR as seir
from funkcije import analysis as analysis
from funkcije import viz


#----------1 DEL---------------#
tip='SF'
oblika='circle'
N=100

G, pos=network.network(tip,oblika, N)



#----------2 DEL---------------#
##Evolucija javnega mnenja
#init parametri za model javnega mnenja#
za=0.1
proti=0.1
nevem=1-za-proti
fleksibilnost=1
rep=1
G=po.opinion(G,za,nevem,proti,rep,fleksibilnost,N)
#Rezultat je mreza, kjer so menja vozlišč določena po principu minimalne energije

#----------3 DEL---------------#

tmaxit=500
beta=0.45
sigma=1/5.1
gamma=1/7
initE=15
dt=1
ponovitev=10
analysis.save_network(G,tip,beta,sigma,gamma,initE)
seir.SEIR(G,ponovitev,tmaxit,beta,sigma,gamma,initE,N,tip,dt)
print("end")

#----------4 DEL---------------#
d,df=analysis.full_analysis()
print(d,df)


#----------5 DEL---------------#
datoteka='SEIR_results/Tip=SF,beta=0.45,sigma=0.20,gamma=0.14,InitE=15.00/pon=_0.csv' #primer datoteke, poljubno
x="1"   #posamezna simulacija

viz.mreža(G,pos,x)
viz.delež(x)
viz.časovni_potek(datoteka)
viz.animacija(datoteka)
viz.porazdelitev(x)
viz.agregacija(x)