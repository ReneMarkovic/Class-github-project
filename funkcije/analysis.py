import numpy as np
import copy
import os
import networkx as nx
import community
from scipy.stats import itemfreq
from collections import Counter


def find_max(location):
	max_list=[]
	for file in os.listdir(location):
		file_path=location+file
		data=np.loadtxt(file_path)
		
		#max_height
		max_value=max(data)

		#time of max height
		max_time=np.where(data==max_value)[0][0]

		#full width at half maximum
		w_left=np.where(data[:,max_time]<max_value/2)[0][-1]
		w_right=np.where(data[max_time_]<max_value/2)[0][0]
		max_width=w_left-w_right

		max_list.append([max_value,max_time,max_width])


	return np.array(max_list)

def frequency_count(bin_width,data):
	freq_list=[]
	min_value=min(data)
	max_value=max(data)
	data=np.array(data)
	val=min_value
	while(val<=max_value+bin_width):
		freq_list.append([val+bin_width/2,len([x for x in data if x >= val and x < val+bin_width])])
		val+=bin_width

	return np.array(freq_list)

def average_trend(location):
    NumberOfFiles=len(os.listdir(location))
    data_length=len(np.loadtxt(os.listdir(location)[0]))
    
    data_matrix=((NumberOfFiles,data_length),float)
    
    #fill matrix
    for i in range(NumberOfFiles):
        data=np.loadtxt(location+"//"+os.listdir(location)[i])
        data_matrix[i,:]=data
        
    #calculate mean and std. deviation
    ret_list=[]
    for i in range(data_length):
        ret_list.append([np.mean(data_matrix[:,i]),np.std(data_matrix[:,i])])
        
    return np.array(ret_list)
    
    
def network_parameters(path):
    graph=nx.read_gexf(path)
    
    #modularity
    part = community.best_partition(graph)
    mod = community.modularity(part,graph)
    
    #clustering
    c=nx.average_clustering(graph,weight='weight')
    
    #distribution
    node_degree=[val for (node, val) in graph.degree()]
    freq = itemfreq(node_degree)
    a = freq[:,0]
    b = freq[:,1]
    
    #shortest path
    for (u,v,d) in graph.edges(data=True):
        d['weight']=1/d['weight']
    l=nx.average_shortest_path_length(graph, weight="weight")
    
    
    return [c, l, mod, a, b]
