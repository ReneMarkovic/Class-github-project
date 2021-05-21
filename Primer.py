import networkx as nx
import random

N=100
G=nx.Graph()

tip='random'

la1=["A","B","C"]

for i in range(N):
    G.add_node(i,A1=random.choice(la1),x=0,y=0)

for i in range(N):
    for j in range(i+1,N,1):
        if random.random()>0.8:
            G.add_edge(i,j,weight=random.random())

nx.write_gexf(G,tip+"_mreza.gexf")
G.clear()

G=nx.read_gexf(tip+"_mreza.gexf")
print(len(G))
for i in G:
    print(i,G.nodes[i]['A1'])
    G.nodes[i]['A1']='Brisi'
    print(i,G.nodes[i]['A1'])
    for j in G:
        if G.has_edge(i,j):
            print(i,j,G[i][j]['weight'])
