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
    
    data_matrix_I_cum=np.zeros((NumberOfFiles,data_length-1),float)
    
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
        
        temp=np.diff(data_I)
        temp[temp<0] = 0
        temp=np.cumsum(temp)
        data_matrix_I_cum[i,:]=temp
        
    #calculate mean and std. deviation
    ret_list_S=[]
    ret_list_E=[]
    ret_list_I=[]
    ret_list_R=[]
    
    ret_list_I_cum=[]
    for i in range(data_length):
        ret_list_S.append([np.mean(data_matrix_S[:,i]),np.std(data_matrix_S[:,i])])
        ret_list_E.append([np.mean(data_matrix_E[:,i]),np.std(data_matrix_E[:,i])])
        ret_list_I.append([np.mean(data_matrix_I[:,i]),np.std(data_matrix_I[:,i])])
        ret_list_R.append([np.mean(data_matrix_R[:,i]),np.std(data_matrix_R[:,i])])
        
    for i in range(data_length-1):
        ret_list_I_cum.append([np.mean(data_matrix_I_cum[:,i]),np.std(data_matrix_I_cum[:,i])])
        
    return [np.array(ret_list_S),np.array(ret_list_E),np.array(ret_list_I),np.array(ret_list_R),np.array(ret_list_I_cum)]
    
    
def network_parameters(location):
    
    graph_file = [f for f in os.listdir(location) if f.endswith('.gexf')]
    graph=nx.read_gexf(location+"//"+graph_file[0])
    tip=graph_file[0].split("_")[0]
    N=graph.number_of_nodes()
    #modularity
    part = community.best_partition(graph)
    mod = community.modularity(part,graph)
    
    #clustering
    c=nx.average_clustering(graph)#,weight='weight')
    
    #distribution
    node_degree=[val for (node, val) in graph.degree()]
    freq = itemfreq(node_degree)
    a_ = freq[:,0]
    b_ = freq[:,1]
    
    #shortest path
    #for (u,v,d) in graph.edges(data=True):
        #d['weight']=1/d['weight']
    l=nx.average_shortest_path_length(graph)#, weight="weight")
    
    #opinion
    f,a,u=0,0,0
    for node in graph:
        if(graph.nodes[node]['atribut']==5):
            f+=1
        if(graph.nodes[node]['atribut']==8):
            u+=1
        if(graph.nodes[node]['atribut']==9):
            a+=1
    
    return [tip,c, l, mod, a_, b_, f/N, a/N, u/N]

def full_analysis():
    rootdir = 'SEIR_results'
    folders=[x[1] for x in os.walk(rootdir)][0]
    
    d={}
    
    df_par=[]
    
    i=0
    for folder in folders:
        key="%d"%(i)
        d[key]={}
        location=rootdir+"//"+folder
        
        #AVG SEIR------------------------------------------------------
        d[key]["avg_SEIR"]={}
        S,E,I,R,I_cum=average_SEIR(location)
        d[key]["avg_SEIR"]["S"]=S[:,0],S[:,1]
        d[key]["avg_SEIR"]["E"]=E[:,0],E[:,1]
        d[key]["avg_SEIR"]["I"]=I[:,0],I[:,1]
        d[key]["avg_SEIR"]["R"]=R[:,0],R[:,1]
        
        d[key]["avg_SEIR"]["I_cum"]=I_cum[:,0],I_cum[:,1]
        
        #INFECTED------------------------------------------------------
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
        
        #NETWORK PARAMETERS
        d[key]["Network"]={}
        tip,c,l,mod,a_,b_,f,a,u=network_parameters(location)
        d[key]["Network"]["type"]=tip
        d[key]["Network"]["clustering"]=c
        d[key]["Network"]["path"]=l
        d[key]["Network"]["modularity"]=mod
        d[key]["Network"]["degree_dist"]=a_,b_
        d[key]["Network"]["opinions"]=f,a,u
        
        #PARAMETERS
        d[key]["Parameters"]={}
        par=np.loadtxt(location+"//"+"parameters.dat")
        d[key]["Parameters"]["beta"]=par[0]
        d[key]["Parameters"]["sigma"]=par[1]
        d[key]["Parameters"]["gamma"]=par[2]
        d[key]["Parameters"]["initE"]=par[3]
        d[key]["Parameters"]["N"]=par[4]
        
        #DF PARAMETERS
        df_par.append([i,tip,par[0],par[1],par[2],par[3],par[4],f,a,u])
        
        i+=1
    
    
    df=pd.DataFrame({"Folder":[x[0] for x in df_par],
                     "Type":[x[1] for x in df_par],
                     "Beta":[x[2] for x in df_par],
                     "Sigma":[x[3] for x in df_par],
                     "Gamma":[x[4] for x in df_par],
                     "InitE":[x[5] for x in df_par],
                     "N":[x[6] for x in df_par],
                     "For":[x[7] for x in df_par],
                     "Against":[x[8] for x in df_par],
                     "Undecided":[x[9] for x in df_par],})
    
    return [d,df]
        
"""      
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
"""