# This Program is used to analyze properties of two-component nanoparticle. It 
# is primarily designed to assist Gromacs analysis 
#    Copyright (C) 2021 Subhamoy Mahajan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The codes used in this software have been used in publications. If you find 
# this useful, please cite:
#
# (1) Subhamoy Mahajan and Tian Tang, "Polyethylenimine-DNA Ratio Strongly 
#     Affects Their Nanoparticle Formation: A Large-Scale Coarse-Grained 
#     Molecular Dynamics Study", 2019, J. Phys. Chem. B, 123 (45), 9629-9640, 
#     DOI: https://doi.org/10.1021/acs.jpcb.9b07031
#

import networkx as nx
import numpy as np
from numba import njit, prange
import copy
import os
import math
from . import geometry

small=1E-6

def get_pdgraph(connected_t):
    """Calculates a mathematical graph containing all DNAs and PEIs as nodes.
       DNA-PEI nodes are connected with an edge, if they are bound together. 
    
    Parameters
    ----------
    connected_t: 2D array of integers
        One timeframe of connected_pickle in connMat.gro2connected().

    Returns
    -------
    pd_graph: a networkx graph
        Contains all DNAs and PEIs as nodes 'd[i]' and 'p[i]' respectively,
        where [i] and [j] are DNA and PEI ID respectively. All IDs begin with 0.
        If DNA [i] is bound with PEI [j], an edge exists between 'd[i]' and 
        'p[j]' nodes. To trace free PEIs, an edge exists between free PEIs and 
        'fp'. Similarly, an edge exists between free DNAs and 'fd'. Free 
        molecules are not bound to any molecule of the other type. 
    """
    const=connected_t.shape
    pd_graph=nx.Graph()
    pd_graph.add_node('fp')
    pd_graph.add_node('fd')
    for d in range(const[0]):
        for p in range(const[1]):
            if connected_t[d,p]:
                # Creates nodes for bound DNA-PEI pair and connects them with an
                # edge.
                pd_graph.add_edge('d'+str(d),'p'+str(p))
    for d in range(const[0]): 
        # If DNA not already present is a free DNA.
        if 'd'+str(d) not in pd_graph: 
            pd_graph.add_edge('d'+str(d),'fd')

    for p in range(const[1]): 
        # If PEI not already present is a free PEI.
        if 'p'+str(p) not in pd_graph: 
            pd_graph.add_edge('p'+str(p),'fp')
    return pd_graph

def plot_pdgraph(connected_t,node_size=500,show=True,filename=None,dpi=300):
    """Plots the pd-graph for a given timestep.
    
    Parameters
    ----------
    connected_t: 2D array of integers
        see get_pdgraph()
    node_size: int, Optional
        Node size for the plot. (Default 500)
    show: bool, Optional
        If true the plot will be shown. If false the plot will be saved as a png
        (Default True)
    filename: str, Optional
        If show is False, the png file will be saved with this filename.
        (Default None)
    dpi: int, Optional
        Dots per inch of the output png file. (Default 300)

    Writes
    ------
    [filename]: PNG file
        If show is false, the pdgraph will be saved as a png file.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.figure(1,figsize=(6,6))

    pd_graph=get_pdgraph(connected_t)
    pos_nodes=nx.drawing.nx_agraph.graphviz_layout(pd_graph,prog='neato') 
    NP_graphs=[pd_graph.subgraph(x) for x in nx.connected_components(pd_graph)]
    for graph in NP_graphs:
        col=[]
        for node in graph:
            if node[0]=='d':
                col.append('r')
            elif node[0]=='p':
                col.append('g')
            else:
                col.append('b')
        nx.draw(graph,pos_nodes,node_size=node_size,node_color=col,with_labels=False)
    if show:
        plt.show()
    elif filename!=None:
        plt.savefig(filename,dpi=dpi)
   

def get_cluster(pd_graph): 
    """Calculates a list of DNA and PEI IDs from pd_graph cluster calculated 
       from get_pdgraph()
   
    Parameters
    ----------
    pd_graph: a networkx graph
        See get_pdgraph() 

    Returns
    -------
    cluster: a 2D list of strings
        cluster[c] contains a list of strings 'd[i]' and 'p[j]', where [i] and 
        [j] are DNA and PEI molecule IDs respectively. [c] is the cluster ID. 
        All IDs begin from 0. The second last row has list of free DNAs along 
        with 'fd', and last row has a list of free PEI along with 'fp'.
    """
    constants=nx.read_gpickle('constants.pickle')
    ndna=constants['ndna']

    cluster=[] #Initialize empty list
    for d in range(ndna):
        if pd_graph.has_edge('d'+str(d),'fd'): # Dont add free DNAs 
            continue
        else:
            clust=[[],[]] # 2D list. index 0 is for DNA, index 1 is for PEI 
            add_clust=1 # Add a new cluster.
            # If the DNA exists in an older cluster don't add it
            for i in range(len(cluster)): 
                if d in cluster[i][0]:
                    add_clust=0 # Dont add cluster
                    break
            if add_clust==1:
                nodes=nx.node_connected_component(pd_graph,'d'+str(d))
                for node in nodes:
                    if node[0]=='d':
                        clust[0].append(int(node[1:]))
                    if node[0]=='p':
                        clust[1].append(int(node[1:]))
                clust[0]=sorted(clust[0])
                clust[1]=sorted(clust[1])
                cluster.append(clust)
    #Second last row for free DNAs.
    freeD_nodes=nx.node_connected_component(pd_graph,'fd')
    freeD_nodes.remove('fd')
    freeD_nodes=[int(x[1:]) for x in freeD_nodes]
    cluster.append([list(sorted(freeD_nodes)),[-1]])
    #Last row for free PEIs.
    freeP_nodes=nx.node_connected_component(pd_graph,'fp')
    freeP_nodes.remove('fp')
    freeP_nodes=[int(x[1:]) for x in freeP_nodes]
    cluster.append([[-1],list(sorted(freeP_nodes))])
    return cluster

def gen_clusters(connected_pickle='connected.pickle', cluster_pickle='cluster.pickle'): 
    """Calculates and pickles clusters data for all timesteps

    It uses function get_pdgraph() and get_cluster() to calculate clusters 
    data in each timestep.

    Parameters
    ----------
    connected_pickle: str, Optional
        See connMat.gro2connected(). (Default 'connected.pickle')
    cluster_pickle: str, Optional
        Filename of the pickled output file. (Default 'cluster.pickle')

    Writes
    ------
    [cluster_pickle].pickle: a pickled file
        Contains a 4D list of cluster information. Axis 0 and 1, represents time
        and cluster ID respectively. For the given cluster ID, the value returned
        is a 2D list of DNA and PEI molecule IDs. The second last cluster ID 
        returns free DNA IDs (molecule ID for PEI is -1 ), and last cluster ID 
        returns free PEI IDs (molecule ID for DNA is -1).
    """
    print("Writing: "+cluster_pickle)
    connected=nx.read_gpickle(connected_pickle)
    times=len(connected)
    clusters=[]
    for t in range(times):
        #Generate cluster for each time step
        pd_graph_t=get_pdgraph(connected[t,:,:]) 
        clusters.append(get_cluster(pd_graph_t))
    nx.write_gpickle(clusters,cluster_pickle) 

def write_cluster(outheader='cluster',cluster_pickle='cluster.pickle',sep=' '): 
    """Writes cluster data into data files, one for each time step.
 
    Parameters
    ---------
    outheader: str, Optional
        the header name of output files. The files [outheader][t].dat will be 
        written, where [t] represents different timesteps. (Default 'cluster')
    cluster_pickle: str, Optional
        See gen_cluster() (Default 'cluster.pickle')
    sep: str, Optional
        A string that separates data. for CSV files use sep=','. (Default ' ')

    Writes
    ------
    [outheader][t].dat: txt file format.
        [t] represents timestep. Each line in the file represents different 
        clusters (or nanoparticles). The lines begin with '[' followed by DNA 
        IDs separated by, followed by '] [', PEI IDs separated by spaces , and 
        finally ']'
    """
    print("Writing: "+outheader+"[t].dat") 
    cluster=nx.read_gpickle(cluster_pickle)
    for t in range(len(cluster)):
        f=open(outheader+str(t)+'.dat','w')
        for cid in range(len(cluster[t])):
            f.write('[ ') # Begining of DNA IDs
            for d in cluster[t][cid][0]: #Write DNA IDs
                f.write(str(d)+sep)
            f.write('] [ ') #End of DNA IDs and begining of PEIs 
            for p in cluster[t][cid][1]:
                f.write(str(p))
                if p!=cluster[t][cid][1][-1]:
                    f.write(sep)
            f.write(' ]\n') #End of the cluster
        f.close() #End of all clusters for time t.

def gen_avgsize(avg_step, outname, main_mol=0, time_pickle='time.pickle', \
    cluster_pickle='cluster.pickle', sep=' '): 
    """Calculates average size of the nanoparticle in terms of number of a main 
       molecule present in the nanoparticle as a function of time.
  
    The main_mol = 0 and 1 represents the main molecule is DNA and PEI 
    respectively.

                              total no. of main molecule in each nanoparticle  
    number average size = -----------------------------------------------------
                                       total no. of clusters

                           squared sum of main molecule in each nanoparticle  
    weight average size = -----------------------------------------------------
                                       total no. of main molecules

    Parameters
    ----------
    avg_step: int
        See connMat.get_roles()
    outname: str
        Output file name containing number and weight average size of the
        nanoparticle
    main_mol: int, Optional
        Chooses a main molecule to calculate size of nanoparticle. 0 represents
        DNA and 1 represents PEI. (Default 0).
    time_pickle: str, Optional
        see connMat.gro2connected()
    cluster_pickle: str, Optional
        See gen_cluster(). (Default 'cluster.pickle')
    sep: str, Optional
        A string that separates data. for CSV files use sep=','. (Default ' ')

    Writes
    ------
    [outname]: txt file format 
        First line is a comment. Each subsequent lines contains time, number 
        average size of the NP averaged over time, and weight average size of
        the nanoparticle averaged over time.
    """
    print("Writing: "+outname) 
    constants=nx.read_gpickle('constants.pickle')
    ndna=constants['ndna']
    npei=constants['npei']
    dna_name=constants['dna_name']
    pei_name=constants['pei_name']

    w=open(outname,"w")
    sim_time=nx.read_gpickle(time_pickle)
    clusters=nx.read_gpickle(cluster_pickle)
    times=len(clusters)
    # N is total number of main molecules in the simulation
    if main_mol==0:
        w.write("# Main molecule is "+dna_name+"\n")
        N=ndna 
    elif main_mol==1: 
        w.write("# Main molecule is "+pei_name+"\n")
        N=npei
    else:
        raise Exception("Incorrect main molecule ID. Only takes 0 or 1.")
       
    w.write('#time'+sep+'number_average_size'+sep+'weight_average_size\n')
    for t in range(times):
        if t%avg_step==0:
            nclust=0  #Number of clusters
            wavg=0  #Weight average cluster size
        #number of main molecules in different clusters at time t.
        #Last two rows are for free DNAs and free PEIs.
        nMains=[len(x[main_mol]) for x in clusters[t]] 
        # Sum of squares + (1^2)* number of free molecules
        wavg+=np.sum(np.square(nMains[:-2]))+nMains[-2+main_mol]
        # Length of nMains ignoring last two rows + number of free molecules
        nclust+=len(nMains)-2+nMains[-2+main_mol]
        if t%avg_step==avg_step-1 or t==times-1:
            #print((t+1)*dt)
            navg=float(N*avg_step)/float(nclust) #Number average size
            wavg=wavg/(N*float(avg_step)) #Weight average size
            tavg=np.average(sim_time[int(t/avg_step)*avg_step:t+1])
            w.write(str(round(tavg,4)) + sep + \
                str(round(navg,4)) + sep + str(round(wavg,4)) + '\n')
    w.close()

def run_ncNP_s(time1, time2, cluster_pickle='cluster.pickle', main_mol=0):
    """Calculates the average number of NPs and charge of NPs asa function of 
       number average size of the nanoparticle. The average is also performed 
       between [time1] to [time2]

    Parameters
    ----------
    time1: int
         time step to start averaing (included).
    time2: int
         time step to end averaging (non included).
    cluster_pickle: str, Optional
        See cluster.gen_cluster(). (Default 'cluster.pickle')
    main_mol: int, Optional
        see gen_avgsize()

    Returns
    ------
    nNP_s: 1D numpy array
        nNP_s[i] contains the average number of nanoparticles with number 
        average size of i+1 (of main molecule).
    cNP_s: 1D numpy array
        nNP_s[i] contains the average charge of nanoparticles with number 
        average size of i+1 (of main molecule).
    """
    constants=nx.read_gpickle('constants.pickle')
    ndna=constants['ndna']
    npei=constants['npei']
    Qdna=constants['Qdna']
    Qpei=constants['Qpei']

    # N is the total number of main molecules in the simulation.
    if main_mol==0:
        N=ndna
    elif main_mol==1:
        N=npei
    else:
        raise Exception("Incorrect main molecule ID. Only takes 0 or 1.")

    nNP_s=np.zeros(N) # Number of NPs for a given size
    cNP_s=np.zeros(N) #Charge of NPs for a given size 
    clusters=nx.read_gpickle(cluster_pickle)
    if time2>len(clusters):
        print("Time2 in run_ncNP_s() exceeds the total number of " \
            "time steps! time2 = total number of timesteps is used")
        time2=len(clusters)-1
    elif time2==len(clusters):
        time2=len(clusters)-1

    for t in range(time1,time2+1):
        for i in range(len(clusters[t])-2):
            ndnas=len(clusters[t][i][0])
            npeis=len(clusters[t][i][1])
            nMains=len(clusters[t][i][main_mol]) 
	    # cNP_s and nNP_s entries are added with nMains-1 because a NP does
            # not exist with 0 main molecules and array index starts with zero. 

            # Charge of NP is charge of PEIs and DNAs in it.
            cNP_s[nMains-1]+=Qpei*npeis+Qdna*ndnas  
            nNP_s[nMains-1]+=1
        if main_mol==0:#DNA
            free_ds=len(clusters[t][-2][0])
            nNP_s[0]+=free_ds
            cNP_s[0]+=free_ds*Qdna
        if main_mol==1:#PEI
            free_ps=len(clusters[t][-1][0])
            nNP_s[0]+=free_ps
            cNP_s[0]+=free_ps*Qpei
    nNP_s=np.array(nNP_s)
    #Divide by total charge of NPs with total number of NPs
    cNP_s=np.divide(cNP_s,np.add(nNP_s,small)) 
    nNP_s=np.divide(nNP_s,(time2-time1+1)) #Average the SDF over time.
    return nNP_s,cNP_s

def gen_ncNP_s(bins, outname, cluster_pickle='cluster.pickle', main_mol=0, \
    sep=' '):
    """Calculates the average number of nanoparticles and charge of 
       nanoparticles over the entire simulation by creating [bins] bins.

    Parameters
    ----------
    bins: int
        Number of bins the entire simulation time is split into to evaluate 
        average. 
    outname: str
        Output file name containing average number of nanoparitcles and charge 
        of nanoparticles calculated o different simulation time-bins
    cluster_pickle: str, Optional
        See cluster.gen_cluster(). (Default 'cluster.pickle')
    main_mol: int, Optional
        see gen_avgsize()
    sep: str, Optional
        A string that separates data. for CSV files use sep=','. 
        (Default ' ')
    
    Writes
    ------
    [outname]: txt file format.
        First line is a comment. Subsequent lines contain 2*[bins]+1 columns. 
        First column is the number average size of the nanoparticle. Next 
        2*[bins] columns contain the average number of nanoparticles and 
        average charge of the nanoparticle evaluated over different bins. The 
        number of rows is equal to the number of DNAs.
    """
    print("Writing: "+outname) 
    constants=nx.read_gpickle('constants.pickle')
    cluster=nx.read_gpickle(cluster_pickle)
    ndna=constants['ndna']
    npei=constants['npei']
    dna_name=constants['dna_name']
    pei_name=constants['pei_name']

    bins=int(bins)
    length=int(len(cluster)/bins)
    w=open(outname,'w')

    # N is the total number of main molecules in the simulation.
    if main_mol==0:
        N=ndna
        w.write("# Main molecule is "+dna_name+"\n")
    elif main_mol==1:
        N=npei
        w.write("# Main molecule is "+pei_name+"\n")
    else:
        raise Exception("Incorrect main molecule ID. Only takes 0 or 1.")
     
    nNP_ss=np.zeros((N,bins))
    cNP_ss=np.zeros((N,bins))
    for i in range(bins):
        nNP_s,cNP_s=run_ncNP_s(i*length,(i+1)*length,cluster_pickle,main_mol)
        nNP_ss[:,i]=nNP_s[:]
        cNP_ss[:,i]=cNP_s[:]

    #Spliting the SDF and CSDF into ''bins'' bins
    head=sep+'number_of_NP'+sep+'charge_of_NP'
    head=head*bins
    w.write('#size'+head+'\n')
    for i in range(N):
        w.write(str(i+1))
        for t in range(bins):
            w.write(sep + str(round(nNP_ss[i][t],4)) + sep + \
                str(round(cNP_ss[i][t],4)))
        w.write('\n')
    w.write('\n')
    w.close()
################################################################################
########################   GRAPH ANALYSIS FUNCTION    ##########################
################################################################################
                
def get_Mgraph(connected_t,main_mol=0): #Test will not be written
    """Calculates a mathematical graph using a list of main molecules. An edge 
       exists between a main molecule pair if they are bridged. 

    Parameters
    ----------
    connected_t: 2D numpy array of bool
        see get_pdgraph()
    main_mol: int, Optional
        see gen_avgsize()

    Returns
    -------
    G: a networkx graph
        Contains nodes '[i]', where [i] is the main molecule ID. If a main 
        molecule pair '[i]' and '[j]' are bridged, an edge exists between 
        the nodes '[i]' and '[j]'.
    """
    #Get Graph from a list of connected DNA (their ids) and the time.
    const=connected_t.shape
    G=nx.Graph()
    for mi in range(const[main_mol]):
        G.add_node(mi)
    for mi in range(const[main_mol]-1):
        for mj in range(mi+1,const[main_mol]):
            if main_mol==0:
                bri=np.dot(connected_t[mi,:],connected_t[mj,:])
            elif main_mol==1:
                bri=np.dot(connected_t[:,mi],connected_t[:,mj])
            if bri>0:
                G.add_edge(mi,mj)
    return G


def graph_bridges(G,connected_t,main_mol=0):
    """ Get bridges in a graph G.

    Parameters
    ----------
    G: networkx Graph()
    connected_t: 2D numpy array of bool
        see get_pdgraph()
    main_mol: int, Optional
        see gen_avgsize()

    Returns
    -------
    bri: Dictionary
        key is tupled (mi,mj) edges and value is the number of bridges.
    """
    bri={}
    for mi,mj in G.edges:
        if main_mol==0:
            bri[(mi,mj)]=np.dot(connected_t[mi,:],connected_t[mj,:])
        elif main_mol==1:
            bri[(mi,mj)]=np.dot(connected_t[:,mi],connected_t[:,mj])
        if bri==0:
            return {}
    return bri
        

def graph_is_equal(G1,G2): #Test written
    """Checks is a mathematical graph has same list of edges and nodes.
    
    Does not check multigraphs. 
    Parameters
    ----------
    G1: A networkx Graph() (1)
    G2: A networkx Graph() (2)
   
    Returns
    -------
    True: If the sorted list of nodes and edges are same. It does not check
          the edge attributes.
    False: I all other cases it returns false.
    """
    for n in G1.nodes:
        if n not in G2:
            return False
    
    for n in G2.nodes:
        if n not in G1:
            return False
    
    for e in G1.edges:
        if not G2.has_edge(e[0],e[1]):
            return False
    
    for e in G2.edges:
        if not G1.has_edge(e[0],e[1]):
            return False
    return True  

def get_UID(G,UniqGs): #Test written
    """Returns the ID of a mathematical graph from a list of unique mathematical
       graphs as reference.
 
    Parameters
    ----------
    G: A networkx graph.
    UniqGs: list of unique networkx Graph()

    Returns
    -------
    i: int
        The index of the graph G in UniqGs.    
    """
    if type(UniqGs) != list:
        raise Exception("Type of UniqGs is not a list. Check order of " + \
            "arguments.")
    for i in range(len(UniqGs)):
        #Checks if the two graphs has the same nodes and edges
        if graph_is_equal(UniqGs[i],G):
            return i
    raise Exception("The graph provided is not in UniqGs. Update UniqGs!")

def update_UniqGs(G,UniqGs): #Test Written
    """Updates the list of unique mathematical graphs UniqGs.

    Parameters
    ----------
    See get_UID()
    """
    if type(UniqGs) != list:
        raise Exception("Type of UniqGs is not a list. Check order of " + \
            "arguments.")
    for n in G.nodes:
        if type(n)!=int:
            raise Exception("Nodes in NSG must be an integer!")

    # Sorts the subgraphs.
    subGraphs=list(nx.connected_components(G))
    order=[]
    i=0
    for c in subGraphs:
        Gi=G.subgraph(c)
        nodes=list(Gi.nodes)
        order.append([len(nodes),-min(nodes),i])
        # number of nodes, - lowest node index, index in subGraphs
        i+=1
    order=sorted(order,reverse=True)
    # This gives a sorted list where subgraphs are arranged from largest
    # to smallest. For subgraphs with same number of nodes, subgraph with
    # smaller node value is chosen first.

    N=len(UniqGs)
    for i in range(len(order)):
    # Iterate over subgraphs in G
        Gi=G.subgraph(subGraphs[order[i][2]])
        add=1 # Initialize to add the subgraph.
        for j in range(N):
            if graph_is_equal(Gi,UniqGs[j]):
                add=0 #Do add the graph Gi
                break
        if add==1:#add the graph if not already present.
            UniqGs.append(Gi) #Note: Items added to UniqGs is added to the 
                              #original variable because it is a list.

def get_changes(G1,G2,Nnode_equal=True): #TestWritten
    """ Get the changes in edges from G1 to G2.

    Parameters
    ----------
    G1: A networkx Graph
    G2: A networkx Graph
    Nnode_equal: Bool, Optional
        If true, it will raise an error if number of nodes in G1 and G2 are 
        not equal. Default True.

    Returns
    -------
    changes: A 2D list
        A sorted list of edges that were either formed or broken.
    """
    if Nnode_equal:
        equal=1
        for n in G1.nodes:
            if n not in G2:
                print(str(n)+' is in G1 but not G2')
                print(sorted(G1.nodes))
                print(sorted(G2.nodes))
                raise Exception("Number of nodes in Graphs are not identical")
        for n in G2.nodes:
            if n not in G1:
                print(str(n)+' is in G2 but not G1')
                print(G1.nodes)
                print(G2.nodes)
                raise Exception("Number of nodes in Graphs are not identical")
    
    changes=[]
    for e in G1.edges: 
        if not G2.has_edge(e[0],e[1]): #broken
            changes.append(list(e))
    for e in G2.edges:
        if not G1.has_edge(e[0],e[1]): #formed
            changes.append(list(e))
    return changes

def has_common_change(L1,L2): #Test written 
    """ Check if change units have common changes.
    
    Parameters
    ----------
    L1: 2D list of changes
    L2: 2D list of changes
        Example. U1->U2  U3->U2 and so on.   

    Returns
    -------
    True: If one element is common
    False: If no element is common
    """
    if len(set(L1[0]) & set(L2[0]))>0:
        return True
    elif len(set(L1[1]) & set(L2[1]))>0:
        return True
    else:
        return False

def remove_loop(GPT, New_NSG_IDs, log=False, wobj=None):
    """ Removes a loop that is formed in the current timestep of a 
        transition graph.

    Parameters
    ----------
    GPT: a networkx DiGraph()
        A graph of principal transitions. Loops, if present, should have only
        been formed in the most recent timestep. All loops that have 
        formed previously shoul have been removed.
    New_NSG_IDs: list of integers
        List of unique IDs of nanoparticle structure graphs (NSG) that are 
        formed in the most recent timestep. 
    log: Bool, Optional
        If True log will be written. (Default False)
    wobj: FILE object, Optional
        file object to write log into. (Default None).

    Returns
    -------
    GPT with removed loops.
    """
    if log: wobj.write('Attempting removal\n')
    if log: wobj.write('------------------\n')
    if log: wobj.write('Current Nodes: '+str(New_NSG_IDs)+'\n') 
    if log: wobj.write('Current edges: '+str(list(GPT.edges(data=True)))+'\n')

    next_NSG_IDs=[]  #List containing Unique IDs that are formed from New_NSG_IDs. 
                     #That demonstrates loops exists.
    if log: wobj.write('New_NSG_IDs: '+str(New_NSG_IDs)+'\n')
    for node in New_NSG_IDs:
        succ=list(GPT.successors(node))
        for ni in succ:
            if ni not in next_NSG_IDs: 
                next_NSG_IDs.append(ni)     
    if log: wobj.write('Next_NSG_IDs: '+str(next_NSG_IDs)+'\n')

    group_NSG_IDs=[] #Group all next_NSG_IDs that are formed simulatenously. 
                     #From the same New_NSG_IDs and at the same time 
                     # (smaller than current time).
    for ni in next_NSG_IDs:
        preds=list(GPT.predecessors(ni))

        # group them by time.
        if len(preds)==0:
            continue
        times=np.array([int(GPT[x][ni]['time']/GPT.graph['dt']+0.5) \
                       for x in preds]) #Time at which next_NSG_IDs are formed
        if log: 
            for i in range(len(preds)):
                wobj.write('\t'+str(preds[i])+' -('+str(times[i]*GPT.graph['dt'])+')-> '+str(ni)+'\n')

        
        uniq=sorted(np.unique(times)) #Sorted unique time index.
        for i in range(len(uniq)):
            foo=[uniq[i],ni]
            idx=np.where(times==uniq[i]) #Items in preds that forms ni 
                                         #at time uniq[i]. 
            for j in range(len(idx[0])):
                foo.append(preds[idx[0][j]])
            group_NSG_IDs.append(foo)       
    #group_NSG_IDs: [ [time_index, next_NSG_ID formed, NSG_IDs producing it ] ,
    #                 ... ]

    #Sort group_NSG_IDs with time in descending order.
    group_NSG_IDs=sorted(group_NSG_IDs,key=lambda x:x[0], reverse=True)
    
    if log: wobj.write('Grouped new transitions='+str(group_NSG_IDs)+' #[time_idx, IDs]\n')
    for nis in group_NSG_IDs:# Remove loops from the most recent formed. 
        if log: wobj.write('Current Node Group: '+str(nis[2:])+'\n')
        GPT_copy=copy.deepcopy(GPT) 
        #Remove all the begining of loop transitions: preds->next_NSG_IDs
        for ni in nis[2:]:
            if GPT_copy.has_edge(ni,nis[1]):
                GPT_copy.remove_edge(ni,nis[1])

        #Check if removing the loops forms a sensible GPT.
        #1. Check if the 
        new_nodes=reachables(GPT_copy)
        removeloop=1
        for ni in New_NSG_IDs:
            if ni not in new_nodes:
                removeloop=0
                break
        if removeloop==1:
            for node in GPT.nodes():
                if node not in GPT_copy: continue
                if node not in new_nodes:
                    GPT_copy.remove_node(node) 
                    if log: wobj.write('Remove node: '+str(node)+'\n')
            GPT=copy.deepcopy(GPT_copy)
            if log: wobj.write('Current nodes: '+str(GPT.nodes)+'\n')
            
    return GPT

def reachables(G): #Test written
    """ Nodes that are reachable from the starting node of the graph.
 
    Parameter
    ---------
    G: A networkx DiGraph()
    
    Returns
    [reach]: a list of nodes that can be reached from starting nodes.
    """
    #reach: nodes that can be reached from starting nodes.
    #initialized as the starting nodes
    reach=copy.deepcopy(G.graph['Snodes']) 
    dt=G.graph['dt']
    tN=int(G.graph['time']/dt+0.5)
    edges={x:[] for x in range(tN+1)}
    for u, v in G.edges():
        edges[int(G[u][v]['time']/dt+0.5)].append([u,v])

    for t in range(tN+1): 
        if len(edges[t])==0:
            continue
        foo={}
        for e in edges[t]:
            if e[1] not in foo:
                foo[e[1]]=[0,0] #number of predecessors, number 
            foo[e[1]][0]+=1
            if e[0] in reach:
                foo[e[1]][1]+=1
            elif type(e[0])==str:
                preds=list(G.predecessors(e[0]))
                succs=list(G.successors(e[0]))
                add=True
                # Check if the connector node as accurate number of edges
                if len(preds)!=G.nodes[e[0]]['num'][0]:
                    add=False
                if len(succs)!=G.nodes[e[0]]['num'][1]:
                    add=False
                # Check if all predecessors are reachable.
                for j in range(len(preds)):
                    if preds[j] not in reach:
                        add=False
                        break
                if add:
                    foo[e[1]][1]+=1
        for x in foo: #For nodes not in reach.
            if foo[x][0]==foo[x][1]: #Check if number of predecessors =
                                     #Number of reachable predecessors 
                if x not in reach:
                    reach.append(x) 
 
    return reach 


def update_GPT(GPT,NSGs_old,NSGs_new,UniqNSGs,time,log=False, dt=1.0, 
    rnd_off=4,  wobj=None): #Not tested
    """ Updates the Graph of principal transitions (GPT) of the nanoparticle 
    structure graphs (NSG).
    
    Nanoparticle structure graphs, are graphs containing only the main 
    molecule as nodes. If two main molecules are bridged, and edge exists 
    between their corresponding nodes.
    
    Each NSG is given an unique ID, and transition in Nanoparticle strucrture 
    is recorded as directed edges between the unique ID (GPT). If all new 
    transitions create loops in GPT, the loops are removed, along with any 
    node (NSG ID) that has no path from the NSG ID's at timestep 0. This retains
    only principal transitions.
    
    GPT nodes: (Unique ID, time) or ('ad','num') or ('re','num')
                'num' is a list [n1, n2]. n1 is the number of edges coming in 
                to a 'ad' or 're' node and n2 is number of edges coming out.
    GPT edges: (Unique ID1, Unique ID2, time)
    GPT.graph={'ad': Number of aggregation dissociation events,
               're': Number of rearrangement events , 
               'Snodes': List of starting nodes,
               'time': The final simulation time for which transitions were 
                       considered,
               'dt': data timestep}

    Parameters
    ----------
    GPT: Directed Networkx Graph
        A GPT graph till previous timestep.
    NSGs_old: Networkx Graph
        Contains all NSGs in the previous timestep
    NSGs_new: Networkx Graph
        Contains all NSGs in the current timestep
    UniqNSGs: List of Networkx Graphs
        List of unique NSGs till the previous timestep.
    time: str or int
        Simulation time.
    log: Bool, Optional
        See remove_loop(). (Default False)
    dt: float
        Stores the dt of GPT if a new GPT is being calculted. If an old GPT 
        is being updated. This value is ignored. (Default 1.0)
    rnd_off: int, Optional
        Number of digits  after decimal. Applied to time. (Default 4)
    wobj: FILE object, Optional 
        See remove_loop(). (Default None)

    Returns
    -------
    Updated GPT
    """
    if  (not isinstance(time,int))  and (not isinstance(time,float)):
        print('time',time)
        raise Exception('Time must be integer or float')
    #Adds new graphs to NSGss.
    update_UniqGs(NSGs_new,UniqNSGs)
    
    New_subGs=[] #New Unique-IDs (UIDs) which are not present in the GPT.
    New_NSG_IDs=[] # New Unique-IDs
    for g in nx.connected_components(NSGs_new):
        Gj=NSGs_new.subgraph(g)
        gid=get_UID(Gj,UniqNSGs)
        New_NSG_IDs.append(gid)
        if gid not in GPT:
            New_subGs.append(gid)
    
    if len(GPT.nodes)==0: #Firt call to this function.
        # Initialize Graph attributes to GPT. 
        # Number of connector nodes.
        # GPT.graph['ad']: Nanoparticle aggregation/dissociation
        # GPT['re']: Nanoparticle redistribution
        
        # GPT['Snodes']: Contains list of starting Nodes.
        # GPT['time']: Contains last simulation. 
        GPT.graph={'ad': 0, 're': 0, 'Snodes': [], 'time': time, 'dt':dt}
        # Add starting nodes to GPT and add it 'Snodes'
        for gid in New_NSG_IDs: 
            GPT.add_node(gid,time=time)
            GPT.graph['Snodes'].append(gid)
        return GPT
    if GPT.graph['time']>=time:
        return GPT # Don't add transitions to the GPT.
    
    GPT.graph['time']=time #Update simulation time        
    if log: wobj.write('\nTime: '+str(round(time,rnd_off))+'\n')
    if log: wobj.write('Current NSGs='+str(New_NSG_IDs)+'\n')
    if log: wobj.write('Current NSGs not in GPT='+str(New_subGs)+'\n')
    # TASK: determine NSGs_old from GPT.

    # If not the first call, then determine changes in the graph
    changes=get_changes(NSGs_old,NSGs_new,False)
    if len(changes)==0: #No change in GPT as is
        return GPT
    
    GPT_changes=[]
    # If there are changes: Determine new transitions  
    for c_edge in changes:
        # change_unit: Group of NSG IDs involved in a single transition. 
        # change_unit[0]: NSG IDs in the previous timestep
        # change_unit[1]: NSG IDs in the current timestep.
        change_unit=[[],[]]
        for ci in nx.connected_components(NSGs_old):
            Gi=NSGs_old.subgraph(ci)
            Gi_id=get_UID(Gi,UniqNSGs) #Unique ID of the 
            if c_edge[0] in Gi or c_edge[1] in Gi: 
            # If nodes in the changed edge (c_edge) is present in subgraph Gi,
            # store the NSG ID in change_unit if not already present.
                if Gi_id not in change_unit[0]:
                    change_unit[0].append(Gi_id)
                        
        for cj in nx.connected_components(NSGs_new):
            Gj=NSGs_new.subgraph(cj)
            Gj_id=get_UID(Gj,UniqNSGs)
            if c_edge[0] in Gj or c_edge[1] in Gj:
            # If nodes in the changed edge (c_edge) is present in subgraph Gj,
            # store the NSG ID in change_unit if not already present
                if Gj_id not in change_unit[1]: 
                    change_unit[1].append(Gj_id)
                    
        GPT_changes.append(change_unit)
        
    #Merge common transitions.
    N=len(GPT_changes)
    for i in range(N-1,0,-1): #N to 1.
        for j in range(i-1,-1,-1): #i-1 to 0
            if has_common_change(GPT_changes[i],GPT_changes[j]):
                #merge changes and remove GPT_changes[i].
                GPT_changes[j][0]=list(set(GPT_changes[i][0])|set(GPT_changes[j][0]))
                GPT_changes[j][1]=list(set(GPT_changes[i][1])|set(GPT_changes[j][1]))
                GPT_changes.pop(i) #Since i>j poping i will not effect the change in indices.
                break

    #Add Transitions     
    for change_unit in GPT_changes: #add transitions
        if len(change_unit[0])==1 and len(change_unit[1])==1:
            #Internal restructuring of NSG
            GPT.add_edge(change_unit[0][0],change_unit[1][0],time=time-dt)
            GPT.nodes[change_unit[1][0]]['time']=time
            if log: wobj.write('Added edge: '+str(change_unit[0][0])+' '+str(change_unit[1][0])+'\n')

        elif len(change_unit[0])==1 and len(change_unit[1])>1:
            #Dissociation of NSG. A connecter 'ad*' is added.
            GPT.add_edge(change_unit[0][0],'ad'+str(GPT.graph['ad']),time=time-dt)
            if log: wobj.write('Added edge: '+str(change_unit[0][0])+' ad'+str(GPT.graph['ad'])+'\n')
            for i in range(len(change_unit[1])):
                GPT.add_edge('ad'+str(GPT.graph['ad']),change_unit[1][i],time=time-dt)
                GPT.nodes[change_unit[1][i]]['time']=time
                if log: wobj.write('Added edge: ad'+str(GPT.graph['ad'])+' '+str(change_unit[1][i])+'\n')
            GPT.nodes['ad'+str(GPT.graph['ad'])]['num']=[1,len(change_unit[1])]
            GPT.graph['ad']+=1
            
        elif len(change_unit[0])>1 and len(change_unit[1])==1:
            #Aggregation of NSGs. A connecter 'ad*' is added.
            for i in range(len(change_unit[0])):
                GPT.add_edge(change_unit[0][i],'ad'+str(GPT.graph['ad']),time=time-dt)
                if log: wobj.write('Added edge: '+str(change_unit[0][i])+' ad'+str(GPT.graph['ad'])+'\n')
            GPT.add_edge('ad'+str(GPT.graph['ad']),change_unit[1][0],time=time-dt)
            GPT.nodes[change_unit[1][0]]['time']=time
            GPT.nodes['ad'+str(GPT.graph['ad'])]['num']=[len(change_unit[0]),1]
            if log: wobj.write('Added edge: ad'+str(GPT.graph['ad'])+' '+str(change_unit[1][0])+'\n')
            GPT.graph['ad']+=1
            
        elif len(change_unit[0])>1 and len(change_unit[1])>1:
            #Redistrubtion of molecules in NSGs. A connecter  'r*' is added.
            for i in range(len(change_unit[0])):
                GPT.add_edge(change_unit[0][i],'re'+str(GPT.graph['re']),time=time-dt)
                if log: wobj.write('Added edge: '+str(change_unit[0][i])+' re'+str(GPT.graph['re'])+'\n')
            for i in range(len(change_unit[1])):
                GPT.add_edge('re'+str(GPT.graph['re']),change_unit[1][i],time=time-dt)
                GPT.nodes[change_unit[1][i]]['time']=time
                if log: wobj.write('Added edge: re'+str(GPT.graph['re'])+' '+str(change_unit[1][i])+'\n')
            GPT.nodes['re'+str(GPT.graph['re'])]['num']=[len(change_unit[0]),len(change_unit[1])]
            GPT.graph['re']+=1

    GPT=remove_loop(GPT,New_NSG_IDs,log,wobj)
    return GPT

def gen_GPT(GPT_in_pickle=None, UniqNSGs_in_pickle = None, main_mol=0, \
    GPT_out_pickle='GPT.pickle', UniqNSGs_out_pickle = 'UniqNSGs.pickle', \
    connected_pickle='connected.pickle', time_pickle = 'time.pickle', log=False,
    logfile='GPT.log',rnd_off=4):#Used
    """ Generate Graph of principal transitions from connection matrix

    Parameters
    ----------
    GPT_in_pickle: str, Optional
        Name of pickled file containing initial GPT. (Default None)
    UniqNSGs_in_pickle: str, Optional
        Name of pickled file containing list of unique NSGs in initial GPT.
        (Default None)
    main_mol: int, Optional
        see cluster.gen_avgsize(). (Default 0)
    GPT_out_pickle: str, Optional
        Name of pickled output file for GPT generated. (Default 'GPT.pickle')
    UniqNSGs_out_pickle: str, Optional
        Name of pickled output file for list of unique NSGs in GPT generated. 
        (Default 'UniNSGs.pickle')
    connected_pickle: str, Optional
        see connMat.gro2connected(). (Default 'connected.pickle')
    time_pickle: str, Optional
        see connMat.gro2connected(). (Default 'time.pickle')
    log: Bool, Optional
        See update_GPT(). (Default False)
    logfile: str, Optional
        File name for logging. (Default 'GPT.log')
    rnd_off: int, Optional 
        See update_GPT()

    Writes
    ------
    [GPT_out_pickle] Updated GPT.
    [UniqNSGs_out_pickle] All unique NSGs.
    [logfile]: For trouble shooting if necessary.    
    """
    if log:
        w=open(logfile,'w')

    GPT=nx.DiGraph()
    NSGs_old=[]
    UniqNSGs=[]
    connected=nx.read_gpickle(connected_pickle)
    sim_time=nx.read_gpickle(time_pickle)
    dt=sim_time[1]-sim_time[0]
    const=connected.shape
    if GPT_in_pickle!=None: #Initialize from pickle if available.
        GPT=nx.read_gpickle(GPT_in_pickle)
        End_UIDs=[]
        # Loops have been removed in saved GPT.
         
        for nodes in GPT:
            if GPT.out_degree(nodes)==0: #TASK: beter algorithm for finding end nodes.
                End_UIDs.append(nodes)

    if UniqNSGs_in_pickle!=None: #Initialize from pickle if avaialble
        UniqNSGs=nx.read_gpickle(UniqNSGs_in_pickle)
        NSGs_old=nx.Graph() #Graph of all NSGs at the last timestep. 
                            #formed from End_UIDs
        for gid in End_UIDs:#Combine graphs of all NSGs
            NSGs_old=nx.compose(NSGs_old,UniqNSGs[gid])
     
    #NOTE: Simulation time is read from time_pickle and compared against time 
    #      stored in GPT

    for t in range(const[0]): 
        NSGs_new=get_Mgraph(connected[t,:,:],main_mol) #NSGs_new has not been added to GPT
        #update_GPT adds NSGs_new to GPT.
        GPT=update_GPT(GPT,NSGs_old,NSGs_new,UniqNSGs,sim_time[t],log=log,dt=dt,
            rnd_off=rnd_off,wobj=w)
        NSGs_old=copy.deepcopy(NSGs_new)
        print("Time: "+str(round(sim_time[t],4)).rjust(6)+' μs',end="\r")
    nx.write_gpickle(GPT,GPT_out_pickle) 
    nx.write_gpickle(UniqNSGs,UniqNSGs_out_pickle)
    if log:
        w.close()

def plot_GPT(GPT_pickle='GPT.pickle', node_size=500,pos_node=None, show=True, \
    edge_label=True, time_pickle='time.pickle'): #Used
    """ Plots Graph of principal transitions 

    Parameters
    ----------
    GPT_pickle: str, Optional
        See GPT_out_pickle in gen_GPT(). (Default 'GPT.pickle')
    node_size: int, Optional
        See plot_pdgraph(). (Default 500)
    pos_node: Dictionary, Optional
        A dictionary with GPT nodes as key and 2D position as value. 
        (Default None)
    show: Bool, Optional
        See plot_pdgraph(). (Default True)
    edge_label: Bool, Optional
        If true time is shown above each arrow in GPT. (Default True)
    time_pickle: str, Optional
        See connMat.gro2connected(). (Default 'time.pickle')
    """
    GPT=nx.read_gpickle(GPT_pickle)
    sim_time=nx.read_gpickle(time_pickle)
    dt=sim_time[1]-sim_time[0]
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,1,figsize=(4,4))
    if pos_node==None:
        pos_nodes=nx.drawing.nx_agraph.graphviz_layout(GPT,prog='neato')
    
    node_label={}
    col=[]
    for node in GPT.nodes:
        a=GPT.in_degree(node)
        b=GPT.out_degree(node)
        if a==1 and b==1:
            node_label[node]=node
        elif a>1 and b==1:
            node_label[node]='+'
        elif a==1 and b>1:
            node_label[node]='-'
        elif a>1 and b>1:
            node_label[node]='re'
        col.append('r')
    
    nx.draw_networkx(GPT,pos_nodes,node_size=node_size,node_color=col,\
        with_labels=True)
    elabel={}
    if edge_label:
        for edge in GPT.edges:
            elabel[(edge[0],edge[1])]=str(round(GPT[edge[0]][edge[1]]['time'],4))
        nx.draw_networkx_edge_labels(GPT,pos_nodes,edge_labels=elabel, font_color='k')
    if show==True:
        for sp in ['top','right','left','bottom']:
            ax.spines[sp].set_visible(False)
        plt.show()
    else:
        plt.savefig('GPT_pickle.png',dpi=1200)
   

def get_weighted_NSGs(wNSG_pickle='wNSG.pickle', GPT_pickle='GPT.pickle', 
    UniqNSGs_pickle='UniqNSGs.pickle', time_pickle='time.pickle', 
    connected_pickle='connected.pickle', main_mol=0, rnd_off=4):
    """
    Calculates Nanoparticle Structure Graphs (NSGs) where the weighted edges is 
    equal to the number of bridging molecules between the main molecule pairs.

    Such weighted NSGs are only calculated for the NSGs present in the Graph of
    principal transitions (GPT). 

    Parameters
    ----------
    wNSG_pickle: str, Optional
        Name of the output pickled file containing weigheted NSGs. 
        (Default 'wNSG.pickle')
    GPT_pickle: str, Optional
        See GPT_out_pickle in gen_GPT(). (Default 'GPT.pickle')
    UniqNSGs_pickle: str, Optional
        See UniqNSGs_out_pickle in gen_GPT(). (Default 'UniqNSGs.pickle')
    time_pickle: str, Optional
        See connMat.gro2connected(). (Default 'time.pickle')
    connected_pickle: str, Optional
        See connMat.gro2connected(). (Default 'connected.pickle')
    main_mol: int, Optional
        See cluster.gen_avgsize(). (Default 0)
    rnd_off: int, Optional
        See update_GPT(). (Default 4)

    Writes
    ------
    wNSG_pickle: pickled file
        A dictionary of weighted NSGs. The key is the unique ID of the NSG (UID)
        as seen in the GPT.
    """
    #Load Pickles
    GPT=nx.read_gpickle(GPT_pickle)
    UniqNSGs=nx.read_gpickle(UniqNSGs_pickle)
    sim_time=nx.read_gpickle(time_pickle)
    connected=nx.read_gpickle(connected_pickle)
    consts=connected.shape
    wNSGs={}
    dt=sim_time[1]-sim_time[0]

    for UID in GPT.nodes:
        if type(UID)==str:
            continue

        print("Working on Node "+str(UID)+" ",end="\n")
        G=UniqNSGs[UID]
        Gnodes=list(G.nodes)
        Gedges=list(G.edges)

        pred_UIDs=list(GPT.predecessors(UID))
        succ_UIDs=list(GPT.successors(UID)) 
        preds_times=[ [x,np.round(GPT[x][UID]['time'],rnd_off)] for x in pred_UIDs ]
        preds_times=sorted(preds_times, key= lambda x: x[1])
        succs_times=[ [x,np.round(GPT[UID][x]['time'],rnd_off)] for x in succ_UIDs ]
        succs_times=sorted(succs_times, key= lambda x: x[1])

        cnt_p=0
        cnt_s=0
        wNSGs[UID]={}
        print('preds_times',preds_times)
        print('succs_times',succs_times)

        while (cnt_p<len(preds_times) and cnt_s<len(succs_times)) or (cnt_p==0 and cnt_s==0):
            time=[]
            next_UIDs_t=[]
            G_t=copy.deepcopy(G)
            add=[1,1]
            if len(preds_times)==0:
                time=[0]
                add[0]=0
            else:
                time=[preds_times[cnt_p][1]+GPT.graph['dt']]
            if len(succs_times)==cnt_s:
                time.append(GPT.graph['time'])
                add[1]=2
            else:
                time.append(succs_times[cnt_s][1])
            time=np.round(time,rnd_off)
            print('Time: '+str(time[0])+'-'+str(time[1]))
            # Calculate average number of bridges between every M-molecule 
            # pairs in the given UID
            cnt=0 #Total number of timesteps UID exists
            for e1,e2 in Gedges:
                G_t[e1][e2]['nb']=0. 

            i1=int(time[0]/dt)
            i2=int(time[1]/dt)
            for i in range(i1,i2+1):
                bris=graph_bridges(G,connected[i,:,:],main_mol)
                for e1, e2 in bris.keys():
                    G_t[e1][e2]['nb']+=bris[(e1,e2)]
                if len(bris.keys())>0:
                    cnt+=1
            print(i1,i2,cnt)
            for e1,e2 in Gedges:
                G_t[e1][e2]['nb']/=float(cnt)

            wNSGs[UID][tuple(np.round(time,rnd_off))]=G_t
            cnt_p+=add[0]
            cnt_s+=add[1]
    nx.write_gpickle(wNSGs,wNSG_pickle)    


def get_cur_next_UIDs(UID, GPT, wNSGs, rnd_off=4):
    """ For a given UID determines the time-range in which it was active
        and current UIDs and next UIDs involved in the transition.

    Parameter
    ---------
    UID: int
        Unique ID of a NSG.
    GPT: a networkx DiGraph() 
        See update_GPT()
    wNSGs: dictionary
        Dictionary of weighted NSG. See get_weighted_NSGS()
    rnd_off: int
        See update_GPT(). (Default 4)
    
    Returns
    -------
    Each UID can potentially have multiple next transitions when GPT contains 
    unremovable loops. First dimension of the lists below iterates over transitions 
    in order of their occurance time. 
 
    cur_UIDs: 2D list
        Second dimension lists the UIDs in the current timestep that are invovled 
        in the transition.
    next_UIDs: 2D list
        Second dimension lists the UIDs in the next timestep that are invovled 
        in the transition.
    H1s: List of networkx Graph()
        Composition of UID Graphs in the current timestep.
    H2s: List of networkx Graph()
        Composition of UID Graphs in the next timestep.
    times: 2D list
        [t0,t1] where cur_UIDs exist from t0-t1 both included.
    """
    times=[]
    cur_UIDs=[]  # List of current nodes.
    H1s=[]
    H2s=[]
    next_UIDs=[]
    pred_UIDs=list(GPT.predecessors(UID))
    succ_UIDs=list(GPT.successors(UID)) 
    preds_times=[ [x,GPT[x][UID]['time']] for x in pred_UIDs ]
    preds_times=sorted(preds_times, key= lambda x: x[1])
    succs_times=[ [x,GPT[UID][x]['time']] for x in succ_UIDs ]
    succs_times=sorted(succs_times, key= lambda x: x[1])

    cnt_p=0
    cnt_s=0
    while (cnt_p<len(preds_times) and cnt_s<len(succs_times)) or (cnt_p==0 and cnt_s==0):
        time=[]
        next_UIDs_t=[]
        add=[1,1]
        if len(preds_times)==0:
            time=[0.]
            add[0]=0
        else:
            time=[preds_times[cnt_p][1]+GPT.graph['dt']]
        if len(succs_times)==cnt_s:
            time.append(GPT.graph['time'])
            xUID=None
            xtime=None
            add[1]=2
        else:
            time.append(succs_times[cnt_s][1])
            xUID=succs_times[cnt_s][0]
            xtime=[np.round(succs_times[cnt_s][1]+GPT.graph['dt'],rnd_off)]

        # Find the time range for next_UID
        if xtime is not None:
            xsucc_UIDs=list(GPT.successors(xUID))
            xsuccs_times=[ [i, np.round(GPT[xUID][i]['time'],rnd_off)] for i in xsucc_UIDs ]
            xsuccs_times=sorted(xsuccs_times, key= lambda x: x[1])
            if len(xsuccs_times)==0:
                xtime.append(np.round(GPT.graph['time'],rnd_off))
            else:
                for i in range(len(xsuccs_times)):
                    if xsuccs_times[i][1]>=xtime[0]:
                        xtime.append(xsuccs_times[i][1])
                        break
                if len(xtime)==1:
                    xtime.append(np.round(GPT.graph['time'],rnd_off))
        cur_UIDs_t=[UID]
        if cnt_s<len(succs_times):
            if type(succs_times[cnt_s][0])==str: #Connector node
                next_UIDs_t=list(GPT.successors(succs_times[cnt_s][0]))
                cur_UIDs_t=list(GPT.predecessors(succs_times[cnt_s][0]))
            else: 
                next_UIDs_t.append(succs_times[cnt_s][0])
                        
        H1_t=nx.Graph()
        for i in range(len(cur_UIDs_t)):
            H1_t=nx.union(H1_t,wNSGs[cur_UIDs_t[i]][tuple(np.round(time,rnd_off))])

        # Combine all NSGs of next node
        H2_t=None
        if len(next_UIDs_t)>0:
            H2_t=nx.Graph()           
            for i in range(len(next_UIDs_t)):
                print(wNSGs[next_UIDs_t[i]])
                H2_t=nx.union(H2_t,wNSGs[next_UIDs_t[i]][tuple(np.round(xtime,rnd_off))])
        cnt_p+=add[0]
        cnt_s+=add[1]

        cur_UIDs.append(cur_UIDs_t)
        next_UIDs.append(next_UIDs_t)
        H1s.append(H1_t)
        H2s.append(H2_t)
        times.append(time)

    return cur_UIDs, next_UIDs, H1s, H2s, times

def get_NSG_pos(G1,G2,UID,NSG_pos,GPT,ref_UIDs=None):
    """ Determine the position of NP sturcture graph for plotting.

    Parameter
    ---------
    G1: networkx Graph()
    G2: networkx Graph()
    UID: int
        See get_cur_next_UIDs()
    NSG_pos: dictionary
        NSG_pos[UID] containst a list of (x,y) Position of all nodes in NSG.
    GPT: networkx DiGraph()
        See update_GPT()
    ref_UIDs: list of int
        List of all UIDs that provide reference positions to current UID.
   
    Return
    ------
    [pos_node]: Dictionary
        Dictionary with NSG nodes as keys and 2D position as values.
    [pos_type]:
        Dictionary with NSG nodes as keys and 0 or 1 as values. If value
        is 0 then position state is unknown. If it is 1 the position state 
        is known and cannot be typically changed.
   
    """
    #pos_nodes: dictionary of (x,y) position of nodes in NSG with UID.
    pos_nodes={}
    #pos_type: dictionary that states poistion are unknown (0) or known (1)
    pos_type={} 
    if UID in NSG_pos: 
        pos_nodes.update(NSG_pos[UID])
    if ref_UIDs!=None:
        for u in ref_UIDs:
            if u in NSG_pos: 
                pos_nodes.update(NSG_pos[u])

    nodes=list(G1.nodes())
    G2nodes=[]
    if G2!=None: 
        G2nodes=list(G2.nodes())
    nodes+=G2nodes

    foo=list(pos_nodes.keys())
    for n in foo:
        if n not in nodes:#n not in G1 or G2
            pos_nodes.pop(n) #pos_nodes contains nodes that are in G1 and G2 and UID.
        else:
            pos_type[n]=1

    nopos_nodes=[] # Stores nodes (DNAs in this case) for which there 
                   # are no positions.
    for n in G1:
        if n not in pos_nodes: 
            nopos_nodes.append(n)
    for n in G2nodes:
        if n not in pos_nodes:
            nopos_nodes.append(n)
    nopos_nodes=list(set(nopos_nodes))

    for n in nopos_nodes:
        pos_type[n]=0

    if len(nopos_nodes)>0:
        foo=list(G1.nodes)+G2nodes
        foo=list(set(foo))
        if len(foo)==len(nopos_nodes): #no positions known
            print('Positions of nodes are unavailable. Generating node ' + \
                  'positions using neato.')
            pos_nodes=nx.drawing.nx_agraph.graphviz_layout(G1,prog='neato')
            min_d2=1E+10
            for u,v in G1.edges():
                r1=pos_nodes[u]
                r2=pos_nodes[v]
                d2=(r1[0]-r2[0])**2+(r1[1]-r2[1])**2
                if min_d2>d2:
                    min_d2=d2
            scale=1/np.sqrt(min_d2)
            if scale>1: 
                scale=1
            for n in pos_nodes:
                pos_nodes[n]=np.array(pos_nodes[n])*scale
                
        else: #some positions known. Unknown nodes must be aggregating.
            print('The following node positions are unavailable: ',nopos_nodes)
            print('Generating random node positions')
            min_d2=1E+10
            for u,v in G1.edges():
                if u in pos_nodes and v in pos_nodes:
                    r1=pos_nodes[u]
                    r2=pos_nodes[v]
                    d2=(r1[0]-r2[0])**2+(r1[1]-r2[1])**2
                    if min_d2>d2:
                        min_d2=d2
            if min_d2>10:
                min_d2=10 

            for i in range(len(nopos_nodes)):
                #Nodes connected to nopos_nodes[i] in the next timestep
                n2=[]
                if G2!=None: 
                    n2=list(G2[nopos_nodes[i]]) 
                th=np.random.random()*np.pi
                dr=np.array([np.cos(th),np.sin(th)])*min_d2
                found=0# Not found the position
                for j in range(len(n2)):
                    if n2[j] in pos_nodes:#Find the position of a connected node.
                        pos_nodes[nopos_nodes[i]]=pos_nodes[n2[j]]+dr
                        found=1 #Found the position
                        break
                if found==0:
                    keys=list(pos_nodes.keys())
                    pos_nodes[nopos_nodes[i]]=dr+pos_nodes[keys[0]]
    return pos_nodes,pos_type


def plot_GPT_NSG_all(GPT_pickle='GPT.pickle', wNSG_pickle='wNSG.pickle',\
    time_pickle='time.pickle', NSG_pos_pickle='NSG_pos.pickle', node_size=600,\
    labels=False,curve_form=None, curve_broken=None, curve_edge=None,\
    log='GPT_NSG.log', time_scale=1., rnd_off=4, figname='GPT-NSG',dpi=200,\
    arrow=False, ref_UIDs=None):
    """ Plot all NSGs present in GPT after their positions have been determined

    Parameter
    ---------
    GPT_pickle: str, Optional
        see get_weighted_NSGs(). (Default 'GPT.pickle')
    wNSG_pickle: str, Optional
        see get_weighted_NSGs(). (Default 'wNSG.pickle')
    time_pickle: str, Optional
        see connMat.gro2connected(). (Default 'time.pickle')    
    NSG_pos_pickle: str, Optional
        see get_NSG_pos(). (Default 'NSG_pos.pickle')
    node_size: int, Optional
        see plot_pdgraph(). (Default 600)
    labels: Bool, Optional
        see plot_graph(). (Default False)
    curve_form: 2D list, Optional
        see plot_graph(). (Default None)
    curve_broken: 2D list, Optional
        see plot_graph(). (Default None)
    curve_edge: 2D list, Optional
        see plot_graph(). (Default None)
    log: str, Optional
        Name of log file. (Default 'GPT_NSG.log')
    time_scale: float, Optional
        Scaling factor for time w.r.t time in pickled time file. (Default 1.)
    rnd_off: int, Optional
        See update_GPT(). (Default 4)
    figname: str, Optional
        Strarting strings for figure output. (Default 'GPT-NSG')
    dpi: int, Optional
        See plot_pdgraph(). (Default 200)
    arrow: Bool, Optional
        If True arrows are plotted to the next NSG with transition time.
        (Default False)
    ref_UIDs: list of ints, Optional
        See get_NSG_pos(). (Default None)

    Writes
    ------
    [figname]_U[UID]_[t0]-[t1].png  : 
        Publication quality NSG plots
    [log]:
        Log file
    Return
    ------  
    """

    #Read_pickles
    import matplotlib.pyplot as plt
    GPT=nx.read_gpickle(GPT_pickle)
    wNSGs=nx.read_gpickle(wNSG_pickle) #?? when is it calculated?
    sim_time=nx.read_gpickle(time_pickle)
    dt=sim_time[1]-sim_time[0]

    NSG_pos={}
    if os.path.isfile(NSG_pos_pickle):
        NSG_pos=nx.read_gpickle(NSG_pos_pickle)

    pos_nodes={}
    Snodes=GPT.graph['Snodes']
    w=open(log,'w')
    finished=[]

    for node in GPT.nodes():
        if type(node)==str: # connector node
            continue
        cur_UIDs, next_UIDs, H1, H2, time=get_cur_next_UIDs(node,  
             GPT, wNSGs)
                

    for snode in Snodes:
        #Add the starting node to the queue
        UID_queue=[snode]
        while len(UID_queue)>0:
            #Get a node from the queue
            UID=UID_queue.pop(0) 
            w.write('UID = '+str(UID)+'\n----------\n')
            cur_UIDs, next_UIDs, H1s, H2s, times=get_cur_next_UIDs(UID, GPT, wNSGs)
            for i in range(len(times)):
                for n in next_UIDs[i]:
                    if n not in finished:
                        if n not in UID_queue: 
                            UID_queue.append(n)
                print('Writing: '+figname+'_U'+str(UID)+'_'+str(round(times[i][0],
                    rnd_off))+'-'+str(round(times[i][1],rnd_off))+'.png')

                w.write('Time: '+str(times[i][0])+'-'+str(times[i][1])+'\n')
                w.write('Current UIDs: '+str(cur_UIDs[i])+'\n')
                w.write('Next UIDs: '+str(next_UIDs[i])+'\n')
                if len(next_UIDs[i])>0:
                    end=False
                    if arrow:
                        arr_time=str(round(times[i][1]*time_scale,rnd_off))
                    else:
                        arr_time=None
                else:
                    end=True
                    arr_time=None

                pos_nodes,stat=get_NSG_pos(H1s[i],H2s[i],UID,NSG_pos,GPT,ref_UIDs=ref_UIDs)

                edge_weight=[] #edge weight
                w.write('Edges:\n')
                for u,v in H1s[i].edges():
                    foo=H1s[i][u][v]['nb']+1
                    edge_weight.append(foo)
                    w.write(str(u)+'-'+str(v)+' weight = '+str(foo)+'\n')
                w.write('\n')
                #formed and broken edge
                formed_edge=[]
                broken_edge=[]
                if not end:
                    changes=get_changes(H1s[i],H2s[i])
                    for edge in changes:
                        if H1s[i].has_edge(edge[0],edge[1]):# edge is broken
                            broken_edge.append([edge[0],edge[1]])
                            w.write('broken edge: '+str(edge[0])+'-'+\
                                str(edge[1])+'\n')
                        else: #edge is formed
                            formed_edge.append([edge[0],edge[1],\
                                H2s[i][edge[0]][edge[1]]['nb']+1])
                            w.write('formed edge: '+str(edge[0])+'-'+\
                                     str(edge[1])+'\n')
                            # node1 node2 wieght of edge.
                w.write('\n')
                fig,ax=plt.subplots(1,1,figsize=(3,3))
                cf, cb, ce = None, None, None
                if curve_form!=None:
                    if UID in curve_form:
                        cf=curve_form[UID]
                if curve_broken!=None:
                    if UID in curve_broken:
                        cb=curve_broken[UID]
                if curve_edge!=None:
                    if UID in curve_edge:
                        ce=curve_edge[UID]
    
                plot_graph(H1s[i],fig,ax,nodepos=pos_nodes,edgeweight=edge_weight,\
                    labels=labels, formed_edge=formed_edge, broken_edge=broken_edge,\
                    curve_form=cf, curve_broken=cb, curve_edge=ce,arrow=arr_time,\
                    log_file=w, nodesize=node_size)

                plt.savefig(figname+'_U'+str(UID)+'_'+str(round(times[i][0],
                    rnd_off))+'-'+str(round(times[i][1],rnd_off))+'.png', dpi=dpi, 
                             bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                finished.append(UID)
    w.close()

def plot_GPT_NSG(UID, GPT_pickle='GPT.pickle', wNSG_pickle='wNSG.pickle',\
    time_pickle='time.pickle', NSG_pos_pickle='NSG_pos.pickle', node_size=600, \
    labels=True,curve_form=None, curve_broken=None, curve_edge=None, \
    arrow=False, time_scale=1, rnd_off=4, ref_UIDs=None):
    """ Plot a NSG in GPT.

    Parameter
    ---------
    UID: int
        UID of NSG to be plotted

    For other parameters, see get_weighted_NSGs()
    """

    #Read_pickles
    import matplotlib.pyplot as plt
    GPT=nx.read_gpickle(GPT_pickle)

    if not GPT.has_node(UID):
        print('GPT does not have the UID '+str(UID))
        return
 
    wNSGs=nx.read_gpickle(wNSG_pickle)
    sim_time=nx.read_gpickle(time_pickle)
    NSG_pos={}
    if os.path.isfile(NSG_pos_pickle):
        NSG_pos=nx.read_gpickle(NSG_pos_pickle)
    pos_nodes={}
    dt=sim_time[1]-sim_time[0]
    cur_UIDs, next_UIDs, H1s, H2s, times=get_cur_next_UIDs(UID,  GPT, wNSGs)
    print('Current UIDs',cur_UIDs)
    print('Next UIDs',next_UIDs)
   
    
    for i in range(len(H1s)):
        time=times[i]
        H1=H1s[i]
        H2=H2s[i]

 
        if time!=None:
            end=False #??
            if arrow:
                arr_time=str(round((time-dt)*time_scale,rnd_off))
            else:
                arr_time=None
        else:
            end=True #??
            arr_time=None

        pos_nodes,stat=get_NSG_pos(H1,H2,UID,NSG_pos,GPT,ref_UIDs=ref_UIDs)
        #Stat is a dictionary. For each node value is 0 if position is randomly generated and 
        # can be changed and is 1 if its obtained from a reference value and cannot be changed
        edge_weight=[H1[u][v]['nb']+1 for u,v in H1.edges()] #edge weight           
        #formed and broken edge
        formed_edge=[]
        broken_edge=[]
        if not end:
            changes=get_changes(H1,H2)
            for edge in changes:
                if H1.has_edge(edge[0],edge[1]):# edge is broken
                    broken_edge.append([edge[0],edge[1]])
                else: #edge is formed
                    formed_edge.append([edge[0],edge[1],\
                                        H2[edge[0]][edge[1]]['nb']+1])
                    # node1 node2 wieght of edge.
        margin=None
        new_pos=pos_nodes
        cnt=0
        while True:
            fig,ax=plt.subplots(1,1,figsize=(3,3))
            plot_graph(H1,fig,ax,nodepos=pos_nodes,edgeweight=edge_weight,\
                labels=labels, formed_edge=formed_edge, broken_edge=broken_edge,\
                curve_form=curve_form,curve_broken=curve_broken,\
                curve_edge=curve_edge,arrow=arr_time, nodesize=node_size, margin=margin )
            plt.show(block=False)
            change=input("Keep existing positions (y/n): ")
            if change=='y':
                NSG_pos[UID]=new_pos
                action=input('Action \nsave: save new positions \n'+ \
                             'scale: Makes the smallest edge length equal to 1 \n'+ \
                             'margin [num]: margin value\n'+ \
                             'q: quit \nEnter: ')
                action=action.split()
                if action[0]=='save':
                    nx.write_gpickle(NSG_pos,NSG_pos_pickle)
                    plt.close()
                    return
                elif action[0]=='scale':
                    edgeL2=1E+6
                    for u,v in H1.edges:
                        r1=pos_nodes[u]
                        r2=pos_nodes[v]
                        d2=(r1[0]-r2[0])**2+(r1[1]-r2[1])**2
                        if edgeL2>d2:
                            edgeL2=d2
                    edgeL=np.sqrt(edgeL2)
                    scale=1/edgeL 
                    for n1 in pos_nodes:
                        pos_nodes[n1]=np.array(pos_nodes[n1])*scale 
                    plt.close()
                    continue
                elif action[0]=='margin':   #This code block does not change anything
                    margin=float(action[1])
                    plt.close()
                    continue
                elif action[0]=='q':
                    return
            elif change=='n':
                if cnt==0:
                    fix=1
                    for key in stat:
                        if stat[key]==1:
                            fix=0
                            break
                    if fix==1:
                        n0=int(input("Fix an initial node. Enter node number: "))
                        stat[n0]=1
                foo=input("Enter nodes. These will be placed CCW in a regular polygon vertex: ")
                ns=foo.split()
                ns=[int(x) for x in ns]
                N=len(ns)
                if cnt==0:
                    r1=pos_nodes[ns[0]]
                    r2=pos_nodes[ns[1]]
                    edgeL=np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)
                rot=float(input("Rotation angle of the polygon in deg: "))
                polygon_pos(len(ns),edgeL,rot,new_pos,ns,stat)
                plt.close()
                cnt+=1

def plot_graph(G, fig, ax, nodepos=None, nodelist=None, edgelist=None, \
    edgeweight=None, labels=False, broken_edge=None, formed_edge=None,\
    curve_form=None, curve_broken=None, curve_edge=None, nodesize=None,\
    arrow=None, log_file=None, margin=None):
    """ Plot a netwrokx graph with some customizations.

    Unknown position is initialized  using networkx 'neato'.

    Parameters
    ----------
    G: netwrokx Graph() or DiGraph()
        Graph to be plotted.
    fig: matplotlib Figure
        Figure to plot the graph
    ax: matplotlib Axes
        Axes to plot the graph
    nodepos: dictionary, Optional
        See pos_node in plot_GPT(). (Default None) 
    nodelist: list, Optional
        List of nodes to be drawn. (Default None)
    edgelist: 2D list, Optional
        List of edges (2 elements) to be drawn. (Default None)
    edgeweight: list, Optional
        List of edge line widths. Has the same order as edgelist. (Default None)
    labels: Bool, Optional
        If True, node names are shown. (Default False)
    broken_edge: 2D list, Optional
        List of edges that get broken in the next timestep. (Default None)
    formed_edge: 2D list, Optional
        List of edges that are formed in the next timestep. (Default None)
    curve_form: 2D list, Optional
        List of edges that are formed in the next timestep, but should be 
        drawn using curved lines. (Default None)
    curve_broken: 2D list, Optional
        List of edges that are get broken in the next timestep, but should be 
        drawn using curved lines. (Default None)
    curve_edge: 2D list, Optional
        List of edges that should be drawn using curved lines. (Default None)
    nodesize: int, Optional
        Scatter size of nodes. (Default None)
    arrow: str, Optional
        Add arrow to the next NSG and the specified string above it. 
        (Default None) 
    log_file: FILE obj, Optional
        File object to write log output. (Default None) 
    margin: float, Optional
        Margin for the plot. If not specified a Default value dependent on NSG size 
        is used. (Default None)
    """

    if nodepos==None:
        nodepos=nx.drawing.nx_agraph.graphviz_layout(G,prog='neato')
    if nodelist==None:
        nodelist=list(G.nodes)
    if edgelist==None:
        edgelist=[]
        for i in range(len(nodelist)-1):
            for j in range(i+1,len(nodelist)):
                if G.has_edge(nodelist[i],nodelist[j]):
                    if curve_edge!=None:
                        add=1
                        for k in range(len(curve_edge)):
                            a=set(curve_edge[k])
                            b=set([nodelist[i],nodelist[j]])
                            if a==b:
                                add=0
                                break
                        if add==1:
                            edgelist.append((nodelist[i],nodelist[j]))
                    else:
                        edgelist.append((nodelist[i],nodelist[j]))
    if edgeweight==None:
        edgeweight=1.0

    if nodesize==None:
        nodesize=300

    cnt=0
    for node in nodelist:
        x,y=nodepos[node]
        if cnt==0:
            min_x,max_x=x,x
            min_y,max_y=y,y
        else:
            if x>max_x:
                max_x=x
            if y>max_y:
                max_y=y
            if x<min_x:
                min_x=x
            if y<min_y:
                min_y=y
        cnt+=1
    dx,dy=[max_x-min_x,max_y-min_y]
    # import matplotlib.pyplot as plt
    nx.draw_networkx(G, ax=ax, pos=nodepos, with_labels=labels, font_color='w', \
        node_color='k',  nodelist=nodelist, node_size=nodesize, \
        edgelist=edgelist, width=edgeweight)
    from matplotlib import patches
    if broken_edge!=None:
        for edge in broken_edge:
            r1=np.array(nodepos[edge[0]])
            r2=np.array(nodepos[edge[1]])
            d=np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)
            rc=(r1+r2)/2.0
            th=math.atan2(r1[1]-r2[1],r1[0]-r2[0])+np.pi/2.0
            foo=np.array([np.cos(th),np.sin(th)])
            fac=[0.15,-0.15]
            if curve_broken!=None:
                for i in range(len(curve_broken)):
                    if set(edge)==set(curve_broken[i]):
                        fac=[0.1-0.25,-0.1-0.25]
                        break
            a1=rc+fac[0]*d*foo
            a2=rc+fac[1]*d*foo
            ax.plot([a1[0],a2[0]],[a1[1],a2[1]],color='r',linewidth=3)
    if formed_edge!=None:
        for edge in formed_edge:
            r1=np.array(nodepos[edge[0]])
            r2=np.array(nodepos[edge[1]])
            if curve_form!=None:
                add=1
                for i in range(len(curve_form)):
                    if set([edge[0],edge[1]])==set(curve_form[i]):
                        add=0
                        cen=0.5*(r1+r2)
                        d=np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)
                        th=math.atan2(r2[1]-r1[1],r2[0]-r1[0])*180/np.pi
                        arc=patches.Arc(cen,d,0.5*d,th,0,180,color='dodgerblue',linewidth=edge[2],zorder=1)
                        ax.add_patch(arc)
                        break
                if add:
                    ax.plot([r1[0],r2[0]],[r1[1],r2[1]],color='dodgerblue',linewidth=edge[2],zorder=1)
            else:
                ax.plot([r1[0],r2[0]],[r1[1],r2[1]],color='dodgerblue',linewidth=edge[2],zorder=1)

    if curve_edge!=None:
        for edge in curve_edge:
            r1=np.array(nodepos[edge[0]])
            r2=np.array(nodepos[edge[1]])
            cen=0.5*(r1+r2)
            d=np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)
            th=math.atan2(r2[1]-r1[1],r2[0]-r1[0])*180/np.pi
            if type(edgeweight)==float:
                arc=patches.Arc(cen,d,0.5*d,th,0,180,color='k',linewidth=1.0,zorder=1)
            else:
                arc=patches.Arc(cen,d,0.5*d,th,0,180,color='k',linewidth=G[edge[0]][edge[1]]['nb']+1,zorder=1)
            ax.add_patch(arc)
       
    #Arrow
    xoff=1
    arr_len=1.8
    h_wid=0.35
    marg_arr=xoff+arr_len
    if arrow==None:
        marg_arr=0

    if margin==None:
        marg=dx*0.15
        if marg<1.0:
            marg=1.0
    else: marg=margin

    ax.set_xlim(-marg+min_x,marg+max_x+marg_arr)
    ax.set_ylim(-marg+min_y,marg+max_y)
    ax.set_aspect('equal',anchor='SW',adjustable='box')
    fig.set_figheight((dy+2*marg)*0.5)
    fig.set_figwidth((dx+2*marg+marg_arr)*0.5)

    #Add arrow
    if arrow!=None:
        ycen=0.5*(max_y+min_y)
        ax.arrow(max_x+xoff, ycen, arr_len, 0, clip_on=False, head_width=h_wid,\
            length_includes_head=True, head_length=2*h_wid, color='k', \
            overhang=0.4, linewidth=2.5)
        ax.text(max_x+xoff+0.5*arr_len, ycen+3*0.5*h_wid, arrow, \
            ha='center', va='bottom', fontsize=25, color='g')
        marg_arr=xoff+arr_len

    if log_file==None:
        print('Xwidth (inch): ',dx+marg,'Y(inch): ',dy+marg,'margin: ',marg)
    else:
        log_file.write('Xwidth (inch): '+str(dx+marg)+'Y(inch): '+\
                               str(dy+marg)+'margin: '+str(marg)+'\n')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
    ax.axis("off")
            

def polygon_pos(n,b,th0,pos,ids,status):
    """ Helps user generate regular polygon position coordinates
        to construct a representation for NSG.

    Position of main molecules with ids, are located on the vertex 
    of a regualar polygon.

    The algorithm to generates position 
    1) according to the polygon if no position is known.
    2) If only one position is known, the polygon is translated
       to match the known position.
    3) If two positions are known, the polgon is scaled and rotated
       to match the two known positions.
    4) If more than two position is known, the results will not be
       desirable. Based on the ids, the first two known positions 
       are matched based on (3). Other known positions are changed.
 
    Parameters
    ----------
    n: int
        number of sides of the polygon
    b: float
        side length of the regular polygon
    th0: float
        polar angle of the first vertex.
    pos: 2D list of floats
        existing known (x,y) positions.
    ids: list of int.
        main mol IDs that are present in the vertex of the polygon.
        the IDs are provided in anti-clockwise order.
    status:
        for each main mol ID, the value is 1 if the position is fixed.
        and 0 if the position can be changed.
    
    Returns
    -------
    """
    th=2*np.pi/float(n) #Angle of rotation for each side.
    r=b/(2*np.sin(th/2.0)) #Distance between center of regular polygon 
                           #and a vertex.
 
    #generate position for polygon
    foo_pos={x:np.zeros(2) for x in ids}
    for i in range(n):
        if ids[i]>=0:
            foo_pos[ids[i]][0]=r*np.cos(th*i+th0*np.pi/180.0)
            foo_pos[ids[i]][1]=r*np.sin(th*i+th0*np.pi/180.0)
    #find existing positions of ids nodes    
    exists=[]
    for key in ids:
        if key<0:
            continue
        if status[key]==1:
            exists.append(key)

    #Works best for only two existing positiosn in ids
    if len(exists)==0:
        for key in ids:
            pos[key]=foo_pos[key]
            status[key]=1
        return pos
    if len(exists)==1:
        
        for key in ids:
            if key==exists[0]:
                continue
            pos[key]=foo_pos[key]+pos[exists[0]]-foo_pos[exists[0]]
    #match a node
    if len(exists)>1:
        dpos=foo_pos[exists[1]]-foo_pos[exists[0]]
        th=math.atan2(dpos[1],dpos[0])
        dpos0=pos[exists[1]]-pos[exists[0]]
        scale=np.sqrt((dpos0[0]**2+dpos0[1]**2)/(dpos[0]**2+dpos[1]**2))
        
        for n in foo_pos:
            foo_pos[n][0]=scale*foo_pos[n][0]
            foo_pos[n][1]=scale*foo_pos[n][1]
        
        th0=math.atan2(dpos0[1],dpos0[0])
        dth=(th0-th)%(2*np.pi)
        c=np.cos(dth)
        s=np.sin(dth)
        RotMat=np.array([[c,-s],[s,c]])
        for key in ids:
            if key==exists[0] or key==exists[1]:
                continue
            dpos=foo_pos[key]-foo_pos[exists[0]]
            new_pos=np.matmul(RotMat,dpos)
            pos[key]=new_pos+pos[exists[0]]
    for key in ids:
        status[key]=1        
    return

def get_connecter_node(GT,sources,destinations):
    """ Get all connector nodes between sources and destinations for a GT.

    GT is a graph of transitions. Unlike GPT all transitions are stored.

    Parameters
    ----------
    GT: a networkx DiGraph()
    sources: ??
        ??
    destinations: ??
        ??

    Returns
    -------
    List of connector nodes or None.
    """
    destinations=set(destinations)
    cnt=0
    #Find all common succesor nodes.
    for ni in sources:
        if ni not in GT:
            return None
        if cnt==0:
            succs=set(GT[ni])
        else:
     
            foo=set(GT[ni])
            succs=succs.intersection(foo)
        cnt+=1
    #Remove all non-connector nodes.
    succs=list(succs)
    for i in range(len(succs)-1,-1,-1):
        if type(succs[i])==int:
            succs.pop(i)

    for i in range(len(succs)):
        dests=set(GT[succs[i]])
        if destinations.issubset(dests):
            return succs[i]
    
    return None
        

def update_GT(GT,NSGs_old,NSGs_new,UniqNSGs,time,dt=1.0):
    """ Updates the Graph of transitions (GT) of the nanoparticle structure 
    graphs (NSG).
    
    Nanoparticle structure graphs, are graphs containing only the main 
    molecule as nodes. If two main molecules are bridged, and edge exists 
    between their corresponding nodes.
    
    Each NSG is given an unique ID, and transition in Nanoparticle strucrture 
    is recorded as directed edges between the unique ID (GT).
    
    Parameters
    ----------
    GT: a networkx DiGraph()
    NSGs_old: Networkx Graph
        See update_GPT()
    NSGs_new: Networkx Graph
        See update_GPT()
    UniqNSGs: List of Networkx Graphs
        See update_GPT()
    time: float
        Simulation time.
    dt: float, Optional
        See update_GPT(). (Default 1.0)
    """
    if  (not isinstance(time,int))  and (not isinstance(time,float)):
        print('time',time)
        raise Exception('Time must be integer or float')

    #Adds new graphs to NSGss.
    update_UniqGs(NSGs_new,UniqNSGs)
    
    New_subGs=[] #New Unique-IDs (UIDs) which are not present in the GT.
    New_NSG_IDs=[] # New Unique-IDs
    for g in nx.connected_components(NSGs_new):
        Gj=NSGs_new.subgraph(g)
        gid=get_UID(Gj,UniqNSGs)
        New_NSG_IDs.append(gid)
        if gid not in GT:
            New_subGs.append(gid)

    if len(GT.nodes)==0: #Firt call to this function.
        # Initialize Graph attributes to GT. 
        # Number of connector nodes.
        # GT.graph['ad']: Nanoparticle aggregation/dissociation
        # GT['re']: Nanoparticle redistribution
        
        # GT['Snodes']: Contains list of starting Nodes.
        # GT['time']: Contains last simulation. 
        GT.graph={'ad': 0, 're': 0, 'time': time, 'Snodes':[]}
        # Add starting nodes to GT and add it 'Snodes'
        for gid in New_NSG_IDs: 
            GT.add_node(gid)
            GT.graph['Snodes'].append(gid)
            nx.set_node_attributes(GT,{gid:[time]},'ts')
        return GT

    if GT.graph['time']>=time:
        return GT # Don't add transitions to the GT.
    
    GT.graph['time']=time #Update simulation time        
    # If not the first call, then determine changes in the graph
    changes=get_changes(NSGs_old,NSGs_new,False)
    if len(changes)==0: #No change in GT as is
        return GT
    
    GT_changes=[]
    # If there are changes: Determine new transitions  
    for c_edge in changes:
        # change_unit: Group of NSG IDs involved in a single transition. 
        # change_unit[0]: NSG IDs in the previous timestep
        # change_unit[1]: NSG IDs in the current timestep.
        change_unit=[[],[]]
        for ci in nx.connected_components(NSGs_old):
            Gi=NSGs_old.subgraph(ci)
            Gi_id=get_UID(Gi,UniqNSGs) #Unique ID of the 
            if c_edge[0] in Gi or c_edge[1] in Gi: 
            # If nodes in the edge is present in subgraph Gi,
            # store the NSG ID in change_unit if not already
            # present.
                if Gi_id not in change_unit[0]:
                    change_unit[0].append(Gi_id)
                        
        for cj in nx.connected_components(NSGs_new):
            Gj=NSGs_new.subgraph(cj)
            Gj_id=get_UID(Gj,UniqNSGs)
            if c_edge[0] in Gj or c_edge[1] in Gj:
            # If nodes in the edge is present in subgraph Gj,
            # store the NSG ID in change_unit if not already
            # present
                if Gj_id not in change_unit[1]: 
                    change_unit[1].append(Gj_id)
                    
        GT_changes.append(change_unit)
        
    #Merge common transitions.
    N=len(GT_changes)
    for i in range(N-1,0,-1): #N to 1.
        for j in range(i-1,-1,-1): #i-1 to 0
            if has_common_change(GT_changes[i],GT_changes[j]):
                #merge changes and remove GT_changes[i].
                GT_changes[j][0]=list(set(GT_changes[i][0])|set(GT_changes[j][0]))
                GT_changes[j][1]=list(set(GT_changes[i][1])|set(GT_changes[j][1]))
                GT_changes.pop(i) #Since i>j poping i will not effect the change in indices.
                break
     
    for change_unit in GT_changes: #add transitions
        if len(change_unit[0])==1 and len(change_unit[1])==1:
            #Internal restructuring of NSG
            #Adding node attributes
            if GT.has_node(change_unit[1][0]):
                GT.nodes[change_unit[1][0]]['ts'].append(time)
            else:
                GT.add_node(change_unit[1][0])
                nx.set_node_attributes(GT,{change_unit[1][0]:[time]},'ts')
            
            if GT.has_edge(change_unit[0][0],change_unit[1][0]):
            #Check if edge exists. Then increase edge weight
                GT[change_unit[0][0]][change_unit[1][0]]['ts'].append(time-dt)
            else:#create new edge with weight 1.
                GT.add_edge(change_unit[0][0],change_unit[1][0],ts=[time-dt])
            
        else:
            #Check if forward transition  exists.
            conn_node=get_connecter_node(GT,change_unit[0],change_unit[1])
            if conn_node!=None:#forward transition exists
                for i in range(len(change_unit[0])):
                    GT[change_unit[0][i]][conn_node]['ts'].append(time-dt)
                    GT.nodes[conn_node]['ts'].append(time) 
                    #forward transisions exist, so connecter node must exist
                for i in range(len(change_unit[1])):
                    GT[conn_node][change_unit[1][i]]['ts'].append(time-dt)
                    GT.nodes[change_unit[1][i]]['ts'].append(time)
                    #forward transisions exist, so change_unit[1][i] must exist
            else:#Check if reverse transition exists
                conn_node=get_connecter_node(GT,change_unit[1],change_unit[0])
                if conn_node==None:#Reverse transition does not exists
                    if len(change_unit[0])==1 or len(change_unit[1])==1: 
                    #Aggregation or dissociation
                        conn_node='ad'+str(GT.graph['ad'])
                        GT.graph['ad']+=1
                    else: #Redistribution
                        conn_node='re'+str(GT.graph['re'])
                        GT.graph['re']+=1
                for i in range(len(change_unit[0])): 
                    if GT.has_node(conn_node):
                        GT.nodes[conn_node]['ts'].append(time)
                    else:
                        GT.add_node(conn_node)
                        nx.set_node_attributes(GT,{conn_node:[time]},'ts')
                    GT.add_edge(change_unit[0][i],conn_node,ts=[time-dt])
                for i in range(len(change_unit[1])):
                    if GT.has_node(change_unit[1][i]):
                        GT.nodes[change_unit[1][i]]['ts'].append(time)
                    else:
                        GT.add_node(change_unit[1][i])
                        nx.set_node_attributes(GT,{change_unit[1][i]:[time]},'ts')
                    GT.add_edge(conn_node,change_unit[1][i],ts=[time-dt])
         
    return GT

def gen_GT(GT_in_pickle=None, UniqNSGs_in_pickle = None, main_mol=0, \
    GT_out_pickle='GT.pickle', UniqNSGs_out_pickle = 'UniqNSGs.pickle', \
    connected_pickle='connected.pickle', time_pickle = 'time.pickle'):
    """ Generate Graph of transitions from connection matrix

    Parameters
    ----------
    GT_in_pickle: str, Optional
        Name of pickled file containing initial GT. (Default None)
    UniqNSGs_in_piklce: str, Optional
        See gen_GPT(). (Default None)
    main_mol: int, Optional
        See gen_avgsize(). (Default 0)
    GT_out_pickle: str, Optional
        Name of pickled output file for GT generated. (Default 'GT.pickle')
    UniqNSGs_out_pickle: str, Optional
        See gen_GPT(). (Default 'UniqNSGs.pickle')
    connected_pickle: str, Optional
        See connMat.gro2connected(). (Default 'connected.pickle')
    time_pickle: str, Optional
        See connMat.gro2connected(). (Default 'time.pickle')
    dt: float, Optional
        See update_GPT(). (Default None)

    Writes
    ------
    [GT_out_pickle] Updated GT.
    [UniqNSGs_out_pickle] All unique NSGs.
    """
    GT=nx.DiGraph()
    NSGs_old=[]
    UniqNSGs=[]
    connected=nx.read_gpickle(connected_pickle)
    sim_time=nx.read_gpickle(time_pickle)
    dt=sim_time[1]-sim_time[0]
    const=connected.shape
    if GT_in_pickle!=None:
        GT=nx.read_gpickle(GT_in_pickle)
        End_UIDs=[]
        for n in GT.nodes():
            if type(n)==str:
                continue
            if abs(GT.nodes[n]['t']-GT.graph['time'])<1E-6:
                End_UIDs.append(n)

    if UniqNSGs_in_pickle!=None:
        UniqNSGs=nx.read_gpickle(UniqNSGs_in_pickle)
        NSGs_old=nx.Graph()
        for gid in End_UIDs:
            NSGs_old=nx.compose(NSGs_old,UniqNSGs[gid])
    
    for t in range(const[0]):
        NSGs_new=get_Mgraph(connected[t,:,:],main_mol)
        GT=update_GT(GT,NSGs_old,NSGs_new,UniqNSGs,sim_time[t],dt=dt)
        NSGs_old=copy.deepcopy(NSGs_new)
        print("Time: "+str(round(sim_time[t],4)).rjust(6)+' μs',end="\r")
    nx.write_gpickle(GT,GT_out_pickle) 
    nx.write_gpickle(UniqNSGs,UniqNSGs_out_pickle)


def plot_GT(GT_pickle='GT.pickle',GPT_pickle='GPT.pickle',time1=0., time2=1., \
    node_pos_pickle='GT_pos.pickle',node_size=500,show=True, show_gt_edge=True,\
    iters=2000,figname='GT.png',dpi=1200):
    """ Plots Graph of principal transitions 

    Parameters
    ----------
    GT_pickle: str, Optional
        see gen_GT(). (Default 'GT.pickle')
    GPT_pickle: str, Optional
        see gen_GPT(). (Default 'GPT.pickle')
    time1: float, Optional
        Starting time to plot GT. (Default 0.)
    time2: float, Optional
        End time to plot GT. (Default 1.)
    node_pos_pickle: str
        Name of pickled file that stores GT nodes as key and their 2D position 
        as value. (Default 'GPT_pos.pickle')
    node_size: int, Optional
        See plot_pdgraph(). (Default 500)
    show: Bool, Optional
        See plot_pdgraph() (Default True)
    show_gt_edge: Bool, Optional
        If true all transition edges that are only present in GT is shown.
        (Default True)
    iters: int, Optional
        Number of iterations to optimize GT node positions. (Default 2000)
    figname: str, Optional
        Output figure name for GT plot. If show is True, figure is not saved.
       (Default 'GT.png')
    dpi: int, Optional
        See plot_pdgraph(). (Default 1200)

    Writes
    ------
    [figname]: GT plot
    """
    GT=nx.read_gpickle(GT_pickle)
    GPT=nx.read_gpickle(GPT_pickle)
    GT_undir=GT.to_undirected()
    T=GT.graph['time']
    import matplotlib.pyplot as plt
    G=get_short_graph(GT,time1,time2)
    plt.figure(1,figsize=(6,6))
    col=[]
    size=[]
    gpt_nodes=[]
    gpt_size=[]
    for node in G.nodes:
        if type(node)==int:#NP structure
            t=G.nodes[node]['ts'][-1]
            #Apply color between blue to red based on time
            foo_col=[(t-time1)/(time2-time1),0,1-(t-time1)/(time2-time1)]
            size.append(node_size)
            if node in GPT: 
                gpt_nodes.append(node)
                #Apply lime green color to nodes inGPT
                foo_col=[50/255,205/255,50/255]
                gpt_size.append(node_size)
            col.append(foo_col)
        elif node[0:2]=='ad': #Aggregation/Dissociation connector
            col.append([0.75,0.75,0.75])
            size.append(0.2*node_size)
        elif node[0:2]=='re':# Multiple NPs exchanging DNAs connector
            col.append([0.25,0.25,0.25])
            size.append(0.2*node_size)
    edgeweight=[np.log(len(G[u][v]['ts'])*0.5+0.5)+1 for u,v in G.edges()] #edge weight 
    gpt_edgelist=[]
    gpt_edgew=[]
    for e in GPT.edges():
        if type(e[0])!=str and type(e[1])!=str:
            if e[0] in G and e[1] in G:
                gpt_edgew.append(np.log(len(G[e[0]][e[1]]['ts'])*0.5+0.5)+1)
                gpt_edgelist.append(e)
    connecter_conv={}
    for n in GPT.nodes():
        #Iterate over connector nodes
        if type(n)==int:
            continue
        n_prev=list(GPT.predecessors(n))
        n_next=list(GPT[n])
        t_gpt=GPT[n][n_next[0]]['time'] 
        #Find 'a' be a connector node in GPT. Then it does not have the same 
        #connecter node name in GPT. So to determine the correct node do the
        #following:
        # (1) Find a previous node 'A' to the current connector node.
        # (2) Find all next nodes to 'A'. One of the connector nodes in the list 
        #     will be the same as the connector node in GPT
        foo=list(G[n_prev[0]])
        conn_node=None
        for i in range(len(foo)):
            # Iterate over connector nodes that are next to 'A'
            if type(foo[i])==int:
                continue
            ts=G[n_prev[0]][foo[i]]['ts']
            for j in range(len(ts)):
                if abs(ts[j]-t_gpt)<1E-6:
                    conn_node=foo[i]
                    break
        if conn_node==None:
            print('Error! Cannot find GPT transitions in GT!')
            exit(-1)
        else:
            connecter_conv[n]=conn_node
            
            for i in range(len(n_prev)):
                gpt_edgelist.append([n_prev[i],conn_node])
                gpt_edgew.append(np.log(len(G[n_prev[i]][conn_node]['ts'])*0.5+0.5)+1)
            for i in range(len(n_next)): 
                gpt_edgelist.append([conn_node,n_next[i]])
                gpt_edgew.append(np.log(len(G[conn_node][n_next[i]]['ts'])*0.5+0.5)+1)

    if os.path.isfile(node_pos_pickle):
        print('Reading: '+node_pos_pickle)
        pos=nx.read_gpickle(node_pos_pickle)
        read=1
    else:
        print('Generating Node positions.')
        pos=nx.networkx.drawing.layout.kamada_kawai_layout(G)
        read=0

    if iters>0:
        print('Iteratively improving node positions using MD_layout()')
        pos_nodes=MD_layout(G,pos,iterations=iters)
        change=1
    else:
        pos_nodes=copy.deepcopy(pos)
        change=0

    if show_gt_edge:
        nx.draw_networkx_edges(G,pos_nodes,width=edgeweight,node_size=size)
    nx.draw_networkx_nodes(G, pos_nodes, node_size=size, node_color=col)
    nx.draw_networkx_edges(G, pos_nodes, edgelist=gpt_edgelist, \
        width=gpt_edgew, edge_color='limegreen',node_size=size,arrowstyle='fancy')
    nx.draw_networkx_nodes(G, pos_nodes,nodelist=gpt_nodes, node_size=gpt_size, node_color='limegreen',edgecolors='darkgreen')
    if read==0 or change==1:    
        print('Writing: '+node_pos_pickle)
        nx.write_gpickle(pos_nodes, node_pos_pickle)
    if show==True:
        plt.show()
    else:
        plt.axis('off')
        print('Writing: '+figname)
        plt.savefig(figname,dpi=dpi, bbox_inches='tight', pad_inches=0)

def get_short_graph(GT,time1,time2):
    """ Returns GT graph between the specified times.

    Parameters
    ----------
    GT: networkx DiGraph()
    time1: float
        Initial time.
    time2: float
        Final time.

    Returns
    -------
    G: networkx DiGraph() 
        A shorter GT graph.
    """

    G=copy.deepcopy(GT)
    #Remove all time instances not between time1, time2
    for node in G:
        N=len(G.nodes[node]['ts'])
        for i in range(N-1,-1,-1):
            t=G.nodes[node]['ts'][i]
            if t<time1 or t>time2:
                G.nodes[node]['ts'].pop(i)
    nodes=list(G.nodes())
    for node in nodes:
        if len(G.nodes[node]['ts'])==0:
            G.remove_node(node)
    
    for e in G.edges():
        N=len(G[e[0]][e[1]]['ts']) 
        for i in range(N-1,-1,-1):
            t=G[e[0]][e[1]]['ts'][i]
            if t<time1 or t>time2:
                G[e[0]][e[1]]['ts'].pop(i)
    edges=list(G.edges())
    for e in edges:
        if len(G[e[0]][e[1]]['ts'])==0:
            G.remove_edge(e[0],e[1])
 
    return G

@njit(parallel=True)
def get_force_energy(pos_old,edges,dist_mat,d0):
    """ Determine force and energy similar to MD simulations.

    Parameters
    ----------
    pos_old: 2D numpy array
        see pos in update_pos()
    edges: 2D list
        List of connected positions.
    disr_mat: 2D numpy array
        Distance matrix from node positions.
    d0: float
        Equilibrium distance between connected nodes.
    """
    N=len(pos_old)
    F=np.zeros((N,2))
    U=0
    for i in prange(N-1):
        for j in prange(i+1,N):
            dr=pos_old[i]-pos_old[j]
            if edges[i,j]>0:#Bonded interaction like force/energy
                foo=edges[i,j]*(dist_mat[i,j]-d0)
                foo_f=foo/dist_mat[i,j]*dr
                F[i]-=foo_f
                F[j]+=foo_f
                U+=foo*foo*0.5
            if dist_mat[i,j]<2.5:
                foo=1/dist_mat[i,j]
                foo2=foo*foo
                foo6=foo2*foo2*foo2
                foo12=foo6*foo6 
                foo_f=(6*foo6-12*foo12)*foo2*dr #Lennard-Jones like force/energy
                F[i]-=foo_f
                F[j]+=foo_f
                U+=(foo12-foo6)
           
    magF=np.zeros(N)
    for i in prange(N):
        magF[i]+=F[i][0]**2+F[i][1]**2
    maxF=np.sqrt(max(magF))
    for i in prange(N):
        F[i]=F[i]/maxF

    return U,F

@njit(parallel=True)
def update_pos(pos,F,step):
    """ Update positions based on displacements and stepsize.

    Parameters
    ----------
    pos: 2D numpy array
        Array of 2D positions.
    F: 2D numpy array
        Array of 2D displacements.
    step: float
        Stepsize to scale the displacements.

    Returns
    -------
    pos: 2D numpy array
        Updated positions.
    """
    pos+=F*step
    return pos 


def MD_layout(G,pos,iterations=100,step=1.,dist0=1.,threshold=1E-6):
    """ A a Graph layout using Molecular Dynamics style energy minimization

    Parameters
    ----------
    G: a networkx DiGraph() or Graph()
    pos: 2D numpy array
        A array of 2D positions.
    iterations: int, Optional
        Number of interations to optimize the Graph layout. (Default 100)
    step: float, Optional
        Stepsize to change node positions. (Default 1.)
    dist0: float, Optional
        Optimal distance between two connected nodes. (Default 1.)
    threshold: float, Optional
        Energy threshold to stop iteration. (Default 1E-6)

    Returns
    -------
    [pos]: Updated Graph node positions.
    """ 
    nodes=list(G.nodes())
    N=len(nodes) 
    #Initial positions
    pos_old=np.zeros((N,2))
    for i in range(N):
        pos_old[i,0]=pos[nodes[i]][0]
        pos_old[i,1]=pos[nodes[i]][1]

    edges=np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1,N):
            ns=[0,0]
            cnts=[0,0]
            if G.has_edge(nodes[i],nodes[j]):
                ns[0]=len(G[nodes[i]][nodes[j]]['ts'])
                cnts[0]+=1
            if G.has_edge(nodes[j],nodes[i]):
                ns[1]=len(G[nodes[j]][nodes[i]]['ts'])
                cnts[1]+=1
            if sum(cnts)==0:
                edges[i,j]=0
            else:
                edges[i,j]=np.log(sum(ns)/sum(cnts)*0.5+0.5)+1

    #Get dist_mat
    dist_mat_old=geometry.dist_matrix(pos_old,maxd=2.51)
    #Get force energy
    Uold,Fold=get_force_energy(pos_old,edges,dist_mat_old,dist0)
    
    cnt=0
    f_cnt=0
    while cnt<iterations:
        #Get positions
        pos_foo=copy.deepcopy(pos_old)
        pos_new=update_pos(pos_foo,Fold,step)
        #Get distmat
        dist_mat_new=geometry.dist_matrix(pos_new,maxd=2.51)
        #Get force energy
        Unew,Fnew=get_force_energy(pos_new,edges,dist_mat_new,dist0)
    
        if Unew<Uold:
            step=1.2*step
            pos_old=copy.deepcopy(pos_new)
            Fold=copy.deepcopy(Fnew)
            dU=Uold-Unew
            Uold=Unew
            cnt+=1
            print('Iteration '+str(cnt)+'     ',end='\r')
            f_cnt=0
            consec_fail=0
            if dU<threshold:
                print('\nThreshold reached')
                break
        else:
            f_cnt+=1
            if f_cnt==100:
                break
            step=0.2*step
    pos={}
    for i in range(N):
        pos[nodes[i]]=pos_old[i]
    return pos 
    



