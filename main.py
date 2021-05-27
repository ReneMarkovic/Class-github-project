import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from funkcije import network

tip='SW'
oblika='circle'
N=100

G, pos=network.network(tip,oblika, N)