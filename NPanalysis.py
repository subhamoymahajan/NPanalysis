# This Program is used to analyze properties of two-component nanoparticle. It is primarily designed to assist Gromacs analysis 
#   Copyright (C) 2021 Subhamoy Mahajan
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
# The codes used in this software have been used in publications. If you find this useful, please cite:
#
# (1) Subhamoy Mahajan and Tian Tang, "Polyethylenimine-DNA Ratio Strongly Affects Their Nanoparticle Formation: A Large-Scale
#     Coarse-Grained Molecular Dynamics Study", 2019, J. Phys. Chem. B, 123 (45), 9629-9640, DOI: https://doi.org/10.1021/acs.jpcb.9b07031
#
#

########################NOTES###############################
# You must run import_constants with the path of constants.py after importing NP_analysis.py
############################################################
import sys
import networkx as nx
import copy
from numpy import zeros
import numpy as np
import math
import re
import glob
small=1E-6

def import_constants(path):
    sys.path.insert(1,path)
    global ndna, npei, contact_dist, Qpei, Qdna, adna, apei, phos_ids, nitr_ids, pei, dna
    from constants import ndna, npei, contact_dist, Qpei, Qdna, adna, apei, phos_ids, nitr_ids, pei , dna

def write_index_mol(outname):
    global adna, apei, ndna, npei
    w=open(outname,"w")
    for d in range(ndna):
        cnt=0
        w.write("[ DNA"+str(d)+" ]\n")
        for di in range(adna):
            w.write(str(d*adna+di+1)+" ")
            cnt+=1
            if cnt%20==0:
                w.write("\n")
        w.write("\n")
    
    for p in range(npei):
        w.write("[ PEI"+str(p)+" ]\n")
        cnt=0
        for pi in range(apei):
            w.write(str(ndna*adna+p*apei+pi+1)+" ") 
            cnt+=1
            if cnt%20==0:
                w.write("\n")
        w.write("\n")
    w.write("\n")

def conv_mindist2connected(times,out_mindist='mindists',out_connected='connected',mindist_loc=''):
    connected=zeros((times,ndna,npei),dtype=int)
    mindists=zeros((times,ndna,npei))
    print('contact_dist = '+str(contact_dist)+' was used')
    for d in range(ndna):
        for p in range(npei):
    	#Read all the mindist files which contains the minimum distance of each pair of DNA-PEIs at different time
            f=open(mindist_loc+"mindist"+str(d)+"-"+str(p)+".xvg","r")
            t=0
            for lines in f:
                foo=lines.split()
                if foo[0][0]=='#' or foo[0][0]=='@':
                    continue
                mindists[t,d,p]=float(foo[1])
                if float(foo[1])<contact_dist and t<times:# Deciding if the minimum distance is below contact distance
                    connected[t,d,p]=1 #0 means not connected, 1 means connected
                t+=1 
            f.close()        
        print("DNA "+str(d)+" done")
    #Pickle Data
    nx.write_gpickle(mindists,out_mindist+'.pickle')
    nx.write_gpickle(connected,out_connected+'.pickle')
   
def get_PEI_roles(avg_step,times,connected_pickle='connected.pickle',outname='PEI_roles.dat'):
    #avg_step = Number of time steps to take an average of number of PEIs in each state.
    connected=nx.read_gpickle(connected_pickle)
    pei_roles=np.sum(connected,axis=1)
    free=np.sum((pei_roles==0),axis=1)
    peri=np.sum((pei_roles==1),axis=1)
    bri=np.sum((pei_roles>1),axis=1)
     
    w=open(outname,"w")
    w.write('# time, free, peripheral, bridging\n ')
    for t in range(int(times/avg_step)):
        w.write(str(avg_step*(t+0.5)/5000)+' '+str(np.average(free[t*avg_step:(t+1)*avg_step]))+' '+str(np.average(peri[t*avg_step:(t+1)*avg_step]))+' '+str(np.average(bri[t*avg_step:(t+1)*avg_step]))+'\n')
    w.close()
    print("pei roles done")


def get_PEI_roles2(avg_step,times,connected_pickle='connected.pickle',outname='PEI_roles2.dat'):
    #avg_step = Number of time steps to take an average.
    connected=nx.read_gpickle(connected_pickle)
    w=open(outname,"w")
    w.write('#time, average number of bridges between a DNA pair, number of DNA pairs\n')
    for t in range(int(times/avg_step)*avg_step+1):
        if t%avg_step==0: #Initialize average to value to zero 
            avg_bri=0 #Average number of bridges between two bound DNAs
            d2d=0 #Average number of DNAs bound to a DNA
        for d1 in range(ndna-1):
            for d2 in range(d1+1,ndna):
                bri=np.dot(connected[t,d1,:],connected[t,d2,:]) #Number of bridging PEI between two DNAs
                if bri>0:
                    avg_bri+=bri
                    d2d+=1
        if t%avg_step==avg_step-1:
            avg_bri/=float(d2d+small)
            d2d/=float(avg_step)
            w.write(str((t+1)/5.0-avg_step/10.0)+' '+str(round(avg_bri,3))+' '+str(round(d2d,3))+'\n')
    w.close()   
    print("pei roles2 done")

def get_pdgraph(connected_t):
    # Connected data type row for dna coloumn for pei
    # Generate Graph
    pd_graph=nx.Graph()
    for d in range(ndna):
        for p in range(npei):
            if connected_t[d,p]:
                pd_graph.add_edge('d'+str(d),'p'+str(p)) #Creates nodes for connecting DNA and PEI pair and connects them with an edge. Examples: Creates nodes for D1 and P20 and connects them.  
        pd_graph.add_node('d'+str(d)) #Creates nodes for DNAs that are not connected to any PEIs
    return pd_graph

def get_cluster(pd_graph): #get cluster from pd_graph
    cluster=[]
    cluster.append(sorted(nx.node_connected_component(pd_graph,'d0'))) #Get all the PEIs and DNAs in the cluster containing  'd0' (including 'd0')
    for d in range(1,ndna):
        con=0#continue if 1
        for i in range(len(cluster)):
            if 'd'+str(d) in cluster[i]:#If a DNA already exists in older clusters continue
                con=1
                continue
        if not con:#If DNA does not exist in older clusters create a new cluster and add all the PEIs and DNAs connected to this DNA.
            cluster.append(sorted(nx.node_connected_component(pd_graph,'d'+str(d))))
    return cluster

def gen_clusters(times,connected_pickle='connected.pickle',headername='cluster'): #writetype = 'file' or 'pickle' 
    connected=nx.read_gpickle(connected_pickle)
    clusters=[]
    for t in range(times):
        #Generate cluster for each time step
        pd_graph_t=get_pdgraph(connected[t,:,:]) 
        C=get_cluster(pd_graph_t)
        clusters_t=[]
        for cid in range(len(C)):
            DNAs=[]
            PEIs=[]
            for i in range(len(C[cid])):
                if C[cid][i][0]=='d':
                    DNAs.append(int(C[cid][i][1:]))
                if C[cid][i][0]=='p':
                    PEIs.append(int(C[cid][i][1:]))
            clusters_t.append([sorted(DNAs),sorted(PEIs)])
        clusters.append(clusters_t)
    nx.write_gpickle(clusters,headername+'.pickle') 
    print("Cluster done")

def gen_avgsize(avg_step,outname,times,dt=0.2,tstart=0,cluster_pickle='cluster.pickle'):
    w=open(outname,"w")
    clusters=nx.read_gpickle(cluster_pickle)

    for t in range(times):
        if t%avg_step==0:
            nclust=0  #Number of clusters
            wavg=0  #Weight average cluster size
        ndnas=[len(x[0]) for x in clusters[t]]
        wavg+=np.sum(np.square(ndnas))
        nclust+=len(ndnas)
        if t%avg_step==avg_step-1:
            #print((t+1)*dt)
            navg=float(ndna*avg_step)/float(nclust) #Number average cluster size =  sum(DNAs)/sum(Clusters)= ndna/nclusters
            wavg=wavg/(ndna*float(avg_step)) #Weight average cluster size = sum(DNA^2)/sum(DNA) = sum(DNA^2)/ndna
            w.write(str(round(tstart+(t+1)*dt - avg_step*dt*0.5,2))+' '+str(round(navg,3))+' '+str(round(wavg,3))+'\n')
    w.close()
    print("avgsize done")

def run_ncNP_s(time1,time2,cluster_pickle='cluster.pickle'):#Generates average number of NPs and charge of NPs for a given size of NP. ??
    nNP_s=np.zeros(ndna) # Number of NPs for a given size
    cNP_s=np.zeros(ndna) #Charge of NPs for a given size 
    clusters=nx.read_gpickle(cluster_pickle)
    for t in range(time1,time2+1):
        for i in range(len(clusters[t])):
            ndnas=len(clusters[t][i][0])
            npeis=len(clusters[t][i][1])
	    #cNP_s and nNP_s entries are added with ndnas-1 because a NP does not exist with 1 DNA and array index starts with zero
            cNP_s[ndnas-1]+=Qpei*npeis+Qdna*ndnas #Charge of NP is charge of PEIs and DNAs in it. 
            nNP_s[ndnas-1]+=1
    nNP_s=np.array(nNP_s)
    cNP_s=np.divide(cNP_s,np.add(nNP_s,small)) #Divide by total charge of NPs with total number of NPs
    nNP_s=np.divide(nNP_s,(time2-time1+1)) #Average the SDF over time.
    return nNP_s,cNP_s

def gen_ncNP_s(times,bins,outname,cluster_pickle='cluster.pickle'):#Generates a average number and charge of NPs for a given size and writes to a file
    #bins= bins total number of time steps in equal portions for taking the average RDF
    global ndna
    str_len0=int(np.log(ndna)/np.log(10))
    str_len=str_len0+8 # 4 for decimals, 1 for period, 3 extra digit space.
    bins=int(bins)
    length=int(times/bins)
    w=open(outname,'w')
     
    nNP_ss=np.zeros((ndna,bins))
    cNP_ss=np.zeros((ndna,bins))
    for i in range(bins):
        nNP_s,cNP_s=run_ncNP_s(i*length,(i+1)*length,cluster_pickle)
        for j in range(ndna):
            nNP_ss[j][i]=nNP_s[j]
            cNP_ss[j][i]=cNP_s[j]

    #Spliting the SDF and CSDF into ''bins'' bins
    w.write('# size, number, charge, number, charge, ...\n')
    for i in range(ndna):
        w.write(str(i+1).rjust(str_len0+1))
        for t in range(bins):
            w.write(str(round(nNP_ss[i][t],4)).rjust(str_len)+str(round(cNP_ss[i][t],4)).rjust(str_len))
        w.write('\n')
    w.write('\n')
    w.close()
    
    print("Avg nNP_s done")

def w2f_cluster(outheader='cluster',cluster_pickle='cluster.pickle'):
    cluster=nx.read_gpickle(cluster_pickle)
    for t in range(len(cluster)):
        f=open(outheader+str(t)+'.dat','w')
        for cid in range(len(cluster[t])):
            f.write('[ ')
            for d in cluster[t][cid][0]:
                f.write(str(d)+' ')
            f.write('] [ ')
            for p in cluster[t][cid][1]:
                f.write(str(p)+' ')
            f.write(']\n')
        f.close()

def read_gro(filename):  #Read a gro file and return position of atoms, box dimensions and important texts in the gro file
    f=open(filename,'r')
    texts=[]
    cnt=0
    for lines in f:
        if cnt==0:
            texts.append(lines)
        elif cnt==1:
            texts.append(lines)
            foo=lines.split()
            Natoms=int(foo[0])
            pos=np.zeros((Natoms,3))
        elif cnt>1 and cnt<Natoms+2:#atoms
            texts.append(lines[:20])
            pos[cnt-2][0]=float(lines[20:28])
            pos[cnt-2][1]=float(lines[28:36])
            pos[cnt-2][2]=float(lines[36:44])
        else:
            texts.append(lines)
            foo=lines.split()
            if len(foo)!=3: #Not supported
                print("Error! code only supported for cubiodal boxes")
                print(foo)
            else:
                box=[float(foo[0]),float(foo[1]),float(foo[2])]
        cnt+=1
    return pos,box,texts

def get_com(pos,typ,indx): #Returns center of mass of the molecule based on the PEI (typ='p') or DNA (typ='d') ID (indx).
    global ndna, adna, apei
    if typ=='d':
        N=adna
        I0=indx*adna
    elif typ=='p':
        N=apei
        I0=ndna*adna+indx*apei
    
    com=[0,0,0]
    for i in range(N):
        for x in range(3):
            com[x]+=pos[I0+i][x]
    com[0]=com[0]/float(N)
    com[1]=com[1]/float(N)
    com[2]=com[2]/float(N)
    return com    

def get_NPcom(pos,dnas,peis):#Returns center of mass of a nanoparticle
    global ndna, adna, apei
    com=[0,0,0]
    for x in range(3):
        for d in dnas:
            for di in range(adna):
                com[x]+=pos[d*adna+di][x]
        for p in peis:
            for pi in range(apei):
                com[x]+=pos[ndna*adna+p*apei+pi][x]
        com[x]=com[x]/float(len(dnas)*adna+len(peis)*apei)
    return com

def get_pbcchange(com1,com2,box): #Returns the change in position required, in terms of the box length, in each direction so that com1, com2 are close.
    #Moving pos2 to pos1
    change=[0,0,0]
    for x in range(3):
        if com2[x]-com1[x]>box[x]/2.0:
            change[x]=-1
        elif com2[x]-com1[x]<-box[x]/2.0:
            change[x]=+1
    return change

def get_pbcchange2(pos,box,d,p,move):#Returns the change in position required by checking the minimum distance
    global ndna, adna, apei
    dmin2=10.0
    min_change=[0,0,0]
    mul=1
    if move=='p':
        mul=-1
    for di in range(adna):
        for pi in range(apei):
            change=[0,0,0]
            d2=0
            for r in range(3):
                foo=pos[d*adna+di][r]-pos[ndna*adna+p*apei+pi][r]
                if foo>box[r]*0.5:
                    d2=d2+(foo-box[r])**2
                    change[r]=-1
                elif foo<-box[r]*0.5:
                    d2=d2+(foo+box[r])**2
                    change[r]=1
                else:
                    d2=d2+foo**2
            if d2<dmin2:
                dmin2=d2
                min_change[0]=change[0]*mul
                min_change[1]=change[1]*mul
                min_change[2]=change[2]*mul
    return min_change,np.sqrt(dmin2)

def change_pos(pos,box,change,typ,indx): #Change positions of a DNA (typ='d')or PEI(typ='p') based on PBC 
    global ndna, adna, apei 
    if typ=='d':
        N=adna
        I0=indx*adna
    elif typ=='p':
        N=apei
        I0=ndna*adna+indx*apei
    for x in range(3):
        if change[x]!=0:
            for i in range(N):
                pos[I0+i][x]+=change[x]*box[x] 
    return pos   

def change_NPpos(pos,box,dnas,peis): #Change positions of a NP using PBC such that center of mass is in the primary box.
    global ndna, adna, apei 
    np_com=get_NPcom(pos,dnas,peis)
    for x in range(3):
        delta=0
        if np_com[x]>2*box[x] or np_com[x]<-2*box[x]:
            print("Error moving center of NP to the box")
        elif np_com[x]>box[x]:
            delta=-box[x]
        elif np_com[x]<0:
            delta=+box[x]
        if delta!=0:
            for d in dnas:
                for di in range(adna):
                    pos[d*adna+di][x]+=delta 
            for p in peis:
                for pi in range(apei):
                    pos[ndna*adna+p*apei+pi][x]+=delta
    return pos

def write_gro(filename,new_pos,box,texts):
    w=open(filename,'w')
    cnt=0
    for i in range(len(texts)):
        w.write(texts[i])
        if cnt>1 and cnt<2+len(new_pos):
            for x in range(3):
                a="{:.3f}".format(new_pos[cnt-2][x])
                w.write(a.rjust(8))
            w.write('\n')
        cnt+=1
    w.close()

def update_gros(times,inname='md_1',outname='New',cluster_pickle='cluster.pickle',connected_pickle='connected.pickle',move_method='com'):
    clusters=nx.read_gpickle(cluster_pickle)
    connected=nx.read_gpickle(connected_pickle)

    for t in range(times): # Iterate over all simulation time
        pos,box,texts=read_gro(inname+str(t)+'.gro')
        pos_new=copy.deepcopy(pos) #Copy positions to change the positions by applying PBC.
        for cid in range(len(clusters[t])): # Iterate over all NPs (revered as clusters)
            finished=[] # Contains all DNAs and PEIs in the NP that have been moved to the right location by applying PBC
            queued=[] #for determining next DNA or PEI to be moved.
            queued.append('d'+str(clusters[t][cid][0][0])) #Adds the first DNA in the NP to the queue
            while len(queued)>0:
                A0=queued.pop(0) #get the first DNA or PEI in the queue and save it as the current molecule
                if A0[0]=='d': #If the current molecule is DNA
                    A0=int(A0[1:]) #A0 is now the DNA ID.
                    nexts=np.where(connected[t,A0,:]==1)[0] #Nexts contains PEI IDs of all PEIs bound to DNA ID A0.

                    if move_method=='com':
                        com1=get_com(pos_new,'d',A0) #Contains center of mass of DNA ID A0
                        #move peis
                        for i in range(len(nexts)): # Iterate over all bound PEIs of DNA ID A0.
                            com2=get_com(pos_new,'p',nexts[i]) #Get center of mass of bound PEIs
                            change=get_pbcchange(com1,com2,box) #Determine the change required  in terms of box length) to make com2 close to com1.
                            pos_new=change_pos(pos_new,box,change,'p',nexts[i]) #update the position of com2    
                    elif move_method=='mindist':
                        for i in range(len(nexts)):
                            change,dmin=get_pbcchange2(pos_new,box,A0,nexts[i],'p') # Determines the change required to move a PEI with ID nexts[i] close to DNA A0
                            if dmin>contact_dist:
                                print('ERROR: dmin',dmin,'contact_dist',contact_dist)
                            pos_new=change_pos(pos_new,box,change,'p',nexts[i]) #update the position of PEI ID 'nexts[i]' 
    
                    finished.append('d'+str(A0)) #Add DNA ID A0 to finished list.
    
                    for i in range(len(nexts)):
                        if 'p'+str(nexts[i]) not in queued: #Add all PEIs bound to DNA ID A0 in queue
                            if 'p'+str(nexts[i]) not in  finished: #Only add PEIs if they are not in the finished list
                                queued.append('p'+str(nexts[i]))
                elif A0[0]=='p': # If the current molecule is PEI
                    A0=int(A0[1:]) #A0 is now the PEI ID
                    nexts=np.where(connected[t,:,A0]==1)[0] #Nexts contains DNA IDs of all DNAs bound to PEI ID A0. 
                    if move_method=='com':
                        com1=get_com(pos_new,'p',A0) #Contains center of mass of PEI ID A0.
                        #move dnas
                        for i in range(len(nexts)):# Iterate over all DNAs bound to PEI ID A0.
                            com2=get_com(pos_new,'d',nexts[i]) #Get center of mass of bound DNA.
                            change=get_pbcchange(com1,com2,box) #Determine the change required (in terms of box length in each direction) to make com2 close to com1.
                            pos_new=change_pos(pos_new,box,change,'d',nexts[i]) #update position of com2  
                    elif move_method=='mindist': 
                        for i in range(len(nexts)):
                            change,dmin=get_pbcchange2(pos_new,box,nexts[i],A0,'d')
                            if dmin>contact_dist:
                                print('ERROR: dmin',dmin,'contact_dist',contact_dist)
                            pos_new=change_pos(pos_new,box,change,'d',nexts[i])

                    finished.append('p'+str(A0)) #Add PEI ID A0 to finished list.
    
                    for i in range(len(nexts)):
                        if 'd'+str(nexts[i]) not in queued: #Add all DNAs bound to PEI ID A0 in queue
                            if 'd'+str(nexts[i]) not in  finished: # Only add DNAs if they are not n the finished list
                                queued.append('d'+str(nexts[i]))
    
            pos_new=change_NPpos(pos_new,box,clusters[t][cid][0],clusters[t][cid][1]) #move the center of mass of the NP to within the primary simulation box.
        write_gro(outname+str(t)+'.gro',pos_new,box,texts) #Write a new .gro file
        print('Time '+str(t)+' Done')

def calc_Rh_Rg(times,Rh_file,Rg_file,mass_file,infile='New',cluster_pickle='cluster.pickle'):
    global ndna, npei, adna, apei
    cluster=nx.read_gpickle(cluster_pickle)
    w=open(Rh_file,'w')
    w1=open(Rg_file,'w')
    Natoms=ndna*adna+npei*apei
    mass=np.zeros(Natoms)
    i=0
    f=open(mass_file,"r")
    for lines in f:
        if i>=Natoms:
            break;
        mass[i]=float(lines[:-1])
        i+=1
    print(mass)
    


    for t in range(times):
        pos,box,text=read_gro(infile+str(t)+'.gro')
        w.write(str(t)+' : ')
        w1.write(str(t)+' : ')
        for cid in range(len(cluster[t])):
            atom_ids=[]
            for d in cluster[t][cid][0]:
                for di in range(adna):        
                    atom_ids.append(d*adna+di)
            for p in cluster[t][cid][1]:
                for pi in range(apei):
                    atom_ids.append(adna*ndna+p*apei+pi)

            #Calculate Rh
            Rhinv=0
            for i in range(len(atom_ids)-1):
                for j in range(i+1,len(atom_ids)):
                    rij=(pos[atom_ids[i]][0]-pos[atom_ids[j]][0])**2+(pos[atom_ids[i]][1]-pos[atom_ids[j]][1])**2+(pos[atom_ids[i]][2]-pos[atom_ids[j]][2])**2
                    Rhinv+=1/np.sqrt(rij)
            Rh=len(atom_ids)**2/Rhinv
            w.write(str(Rh)+' ')    

           #Calculate com
            com=[0,0,0]
            Mtot=0
            for j in range(3):
                for i in range(len(atom_ids)):
                    com[j]+=pos[atom_ids[i]][j]*mass[atom_ids[i]]
                    if j==0:
                        Mtot+=mass[atom_ids[i]]
                com[j]=com[j]/Mtot

           #Calculate Rg
            Rg2=0
            for i in range(len(atom_ids)):
                for j in range(3):
                    Rg2+=mass[atom_ids[i]]*(pos[atom_ids[i]][j]-com[j])**2
            Rg2=Rg2/Mtot
            Rg=np.sqrt(Rg2)
            w1.write(str(Rg)+' ')
        print("Time "+str(t)+" done")

        w.write('\n')
        w1.write('\n')

