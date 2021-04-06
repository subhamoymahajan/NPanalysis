# This Program is used to analyze properties of two-component nanoparticle. It 
# is primarily designed to assist Gromacs analysis 
#    Copyright (C) 2021 Subhamoy Mahajan <subhamoygithub@gmail.com>
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
small=1E-6

def get_pdgraph(connected_t):
    """Calculates a mathematical graph containing all DNAs and PEIs as nodes.
       DNA-PEI nodes are connected with an edge, if they are bound together. 
    
    Parameters
    ----------
    connected_t: 2D array of integers
        Connection matrix at a specific time step. Axis 0 and 1 is for DNA and
        PEI respectively. Value is 0 if the DNA-PEI pair is not connected, and 
        1 if the pair is connected.

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

def plot_pdgraph(connected_t,ns=500,show=True,filename=None,dpi=300):
    """Plots the pd-graph for a given timestep.
    
    Parameters
    ----------
    connected_t: 2D array of integers
        Connection matrix at a specific time step. Axis 0 and 1 is for DNA and
        PEI respectively. Value is 0 if the DNA-PEI pair is not connected, and 
        1 if the pair is connected.
    ns: int, optional
        Node size for the plot. (default value is 500)
    show: bool, optional
        If true the plot will be shown. If false the plot will be saved as a png
        (default is True)
    filename: str, optional
        If show is False, the png file will be saved with this filename.
        (default is None)
    dpi: int, optional
        Dots per inch of the output png file. (default is 300)

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
        nx.draw(graph,pos_nodes,node_size=ns,node_color=col,with_labels=False)
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
        A mathematical graph describing DNA-PEI connections. See 
        cluster.get_pdgraph() for more details.

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
    connected_pickle: str, optional
        Filename which contains pickled connection matrix data. For more details
        see connMat.gro2connected(). (default value is 'connected.pickle')
    out_pickle: str, optional
        Filename of the pickled output file. (default value is 'cluster.pickle')

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
    outheader: str, optional
        the header name of output files. The files [outheader][t].dat will be 
        written, where [t] represents different timesteps. 
        (default value is 'cluster')
    cluster_pickle: str, optional
        Filename which contains pickled cluster data. See cluster.gen_cluster()
        for more details. (default value is 'cluster.pickle')
    sep: str, optional
        A string that separates data. for CSV files use sep=','. 
        (default value is ' ')

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
        Number of timesteps over which the average is evaluated
    outname: str
        Output file name containing number and weight average size of the
        nanoparticle
    main_mol: int, optional
        Chooses a main molecule to calculate size of nanoparticle. 0 represents
        DNA and 1 represents PEI. (default value is 0).
    time_pickle: str, optional

    cluster_pickle: str, optional
        Filename which contains pickled cluster data. See cluster.gen_cluster()
        for more details. (default value is 'cluster.pickle')
    sep: str, optional
        A string that separates data. for CSV files use sep=','. 
        (default value is ' ')

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
        if t%avg_step==avg_step-1:
            #print((t+1)*dt)
            navg=float(N*avg_step)/float(nclust) #Number average size
            wavg=wavg/(N*float(avg_step)) #Weight average size
            tavg=0.5*(sim_time[t+1-avg_step]+sim_time[t+1])
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
    cluster_pickle: str, optional
        Filename which contains pickled cluster data. See cluster.gen_cluster()
        for more details. (default value is 'cluster.pickle')
    main_mol: int, optional
        Chooses a main molecule to calculate size of nanoparticle. 0 represents
        DNA and 1 represents PEI. (default value is 0).

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
    cluster_pickle: str, optional
        Filename which contains pickled cluster data. See cluster.gen_cluster()
        for more details. (default value is 'cluster.pickle')
    main_mol: int, optional
        Chooses a main molecule to calculate size of nanoparticle. 0 represents
        DNA and 1 represents PEI. (default value is 0).
    sep: str, optional
        A string that separates data. for CSV files use sep=','. 
        (default value is ' ')
    
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
