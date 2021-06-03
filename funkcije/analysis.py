import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import pandas as pd
import networkx as nx
import community
from scipy.stats import itemfreq
from collections import Counter


def Infected(location):
    SEIR_files = [f for f in os.listdir(location) if f.endswith('.csv')]
    max_list=[]
    for file in SEIR_files:
        df=pd.read_csv(location+"//"+file)
        
        #max_height
        max_value=max(df["I"])
        
		#time of max height
        max_time=np.where(df["I"]==max_value)[0][0]
        max_day=df["Time"][max_time]
        
        #end of pandemic
        end_time=np.where(df["I"]>0)[0][-1]
        end_day=df["Time"][end_time]
        max_list.append([max_value,max_day,end_day])
    
    #speed of infection
    return np.array(max_list)


def frequency_count(data):
    freq_list=[]
    min_value=min(data)
    max_value=max(data)
    bin_width=(max_value-min_value)/50
    data=np.array(data)
    val=min_value
    while(val<=max_value+bin_width):
        freq_list.append([val+bin_width/2,len([x for x in data if x >= val and x < val+bin_width])])
        val+=bin_width
    return np.array(freq_list)

def average_SEIR(location):
    
    SEIR_files = [f for f in os.listdir(location) if f.endswith('.csv')]
    NumberOfFiles=len(SEIR_files)
    data_length=len(pd.read_csv(location+"//"+SEIR_files[0]))
    
    data_matrix_S=np.zeros((NumberOfFiles,data_length),float)
    data_matrix_E=np.zeros((NumberOfFiles,data_length),float)
    data_matrix_I=np.zeros((NumberOfFiles,data_length),float)
    data_matrix_R=np.zeros((NumberOfFiles,data_length),float)
    
    #fill matrix
    for i in range(NumberOfFiles):
        df=pd.read_csv(location+"//"+SEIR_files[i])
        
        data_S=df["S"]
        data_matrix_S[i,:]=data_S
        
        data_E=df["E"]
        data_matrix_E[i,:]=data_E
        
        data_I=df["I"]
        data_matrix_I[i,:]=data_I
        
        data_R=df["R"]
        data_matrix_R[i,:]=data_R
        
    #calculate mean and std. deviation
    ret_list_S=[]
    ret_list_E=[]
    ret_list_I=[]
    ret_list_R=[]
    for i in range(data_length):
        ret_list_S.append([np.mean(data_matrix_S[:,i]),np.std(data_matrix_S[:,i])])
        ret_list_E.append([np.mean(data_matrix_E[:,i]),np.std(data_matrix_E[:,i])])
        ret_list_I.append([np.mean(data_matrix_I[:,i]),np.std(data_matrix_I[:,i])])
        ret_list_R.append([np.mean(data_matrix_R[:,i]),np.std(data_matrix_R[:,i])])
        
    return [np.array(ret_list_S),np.array(ret_list_E),np.array(ret_list_I),np.array(ret_list_R)]
    
    
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

def full_analysis():
    rootdir = 'SEIR_results'
    folders=[x[1] for x in os.walk(rootdir)][0]
    
    d={}
    i=0
    for folder in folders:
        key="%d"%(i)
        d[key]={}
        location=rootdir+"//"+folder
        
        d[key]["avg_SEIR"]={}
        S,E,I,R=average_SEIR(location)
        d[key]["avg_SEIR"]["S"]=S[:,0],S[:,1]
        d[key]["avg_SEIR"]["E"]=E[:,0],E[:,1]
        d[key]["avg_SEIR"]["I"]=I[:,0],I[:,1]
        d[key]["avg_SEIR"]["R"]=R[:,0],R[:,1]
        
        d[key]["Infected"]={}
        I_array=Infected(location)
        
        d[key]["Infected"]["max_I"]={}
        d[key]["Infected"]["max_I"]["values"]=I_array[:,0]
        d[key]["Infected"]["max_I"]["mean"]=np.mean(I_array[:,0])
        d[key]["Infected"]["max_I"]["std"]=np.std(I_array[:,0])
        d[key]["Infected"]["max_I"]["freq"]=frequency_count(I_array[:,0])[:,0],frequency_count(I_array[:,0])[:,1]
        
        d[key]["Infected"]["max_day"]={}
        d[key]["Infected"]["max_day"]["values"]=I_array[:,1]
        d[key]["Infected"]["max_day"]["mean"]=np.mean(I_array[:,1])
        d[key]["Infected"]["max_day"]["std"]=np.std(I_array[:,1])
        d[key]["Infected"]["max_day"]["freq"]=frequency_count(I_array[:,1])[:,0],frequency_count(I_array[:,1])[:,1]
        
        d[key]["Infected"]["ep_length"]={}
        d[key]["Infected"]["ep_length"]["values"]=I_array[:,2]
        d[key]["Infected"]["ep_length"]["mean"]=np.mean(I_array[:,2])
        d[key]["Infected"]["ep_length"]["std"]=np.std(I_array[:,2])
        d[key]["Infected"]["ep_length"]["freq"]=frequency_count(I_array[:,2])[:,0],frequency_count(I_array[:,2])[:,1]
        
        i+=1
    
    return d
        
        
df=pd.read_csv("SEIR_results//test//pon=_9.csv")
print(len(df))
print(df["Time"][1])
data=np.array(df["I"])


def window(size):
    return np.ones(size)/float(size)

plt.plot(data)
plt.plot(np.convolve(data,window(300),'same'),'r')
plt.show()
print(len(data),len(np.convolve(data,window(50),'same')))

rootdir = 'SEIR_results'
i=0
list_=[x[1] for x in os.walk(rootdir)][0]
print(list_) 

d=full_analysis()
print(d["1"]["Infected"]["max_I"]["freq"][0])

#plt.scatter(d["1"]["Infected"]["max_I"]["freq"][0],d["1"]["Infected"]["max_I"]["freq"][1])
#plt.hist(d["1"]["Infected"]["ep_length"]["values"],10)
#plt.plot(d["1"]["avg_SEIR"]["S"][0])
#plt.plot(d["1"]["avg_SEIR"]["E"][0])
plt.errorbar(df["Iter"],d["1"]["avg_SEIR"]["I"][0],d["1"]["avg_SEIR"]["I"][1])
plt.show()
