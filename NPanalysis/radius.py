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

import os
import networkx as nx
import numpy as np
from numba import njit, prange
from . import gmx

def calc_Rh_t(pos,clust_atoms):
    """Calculates hydrodynamic radius for nanoparticles at a given timestep. 

    It assumes that the group of atoms in each cluster are part of a 
    nanoparticle. It does not apply periodic boundary condition
    
    Parameters
    ----------
    pos: 2D numpy array of floats
        Contains position of all atoms in the system. Axis 0 is the global atom
        ID, and axis 1 is the direction.
    clust_atoms: 2D List of integers
        Contains global ID of atoms in the nanoparticle in each cluster. Axis 0
        is the cluster ID.

    Returns
    -------
    Rhs: float
        Hydrodynamic radius of nanoparticles at a given timestep. 
    """
    Rhs=[]
    for i in range(len(clust_atoms)):
        Rhs.append(calc_Rh(pos,clust_atoms[i]))
    return Rhs

def calc_Rg2_t(pos,clust_atoms,mass):
    """Calculates square of radius of gyration for nanoparticles at a given 
       timestep

    It assumes that the group of atoms in each cluster are part of a 
    nanoparticle. It does not assume periodic boundary condition
    
    Parameters
    ----------
    pos: 2D numpy array of floats
        Contains position of all atoms in the system. Axis 0 is the global atom
        ID, and axis 1 is the direction.
    clust_atoms: 2D List of integers
        Contains global ID of atoms in the nanoparticle in each cluster. Axis 0
        is the cluster ID.
    mass: List of floats
        Contains mass information of all atoms.

    Returns
    -------
    Rg2s: float
        Radius of gyration of nanoparticles at a given timestep.
    """
    Rg2s=[]
    for i in range(len(clust_atoms)):
        Rg2s.append(calc_Rg2(pos,clust_atoms[i],mass))
    return Rg2s

@njit(parallel=True)
def calc_Rh(pos,atoms):
    """Calculates hydrodynamic radius for a given set of atoms. 

    It assumes that the group of atoms are part of a nanoparticle. It 
    does not apply periodic boundary condition
    
    Parameters
    ----------
    pos: 2D numpy array of floats
        Contains position of all atoms in the system. Axis 0 is the global atom
        ID, and axis 1 is the direction.
    atoms: List of integers
        Contains global ID of atoms in the nanoparticle

    Returns
    -------
    Rh: float
        Hydrodynamic radius of a group of atoms (nanoparticle) 
    """
    Rhinv=0
    for i in prange(len(atoms)-1):
        for j in prange(i+1,len(atoms)):
            dif=pos[atoms[i],:]-pos[atoms[j],:]
            rij=np.sqrt(np.dot(dif,dif))
            Rhinv+=1/rij
    Rh=len(atoms)**2/Rhinv

    return Rh

@njit(parallel=True)
def calc_Rg2(pos,atoms,mass):
    """Calculates radius of gyration for a given set of atoms. 

    It assumes that the group of atoms are part of a nanoparticle. It 
    does not assume periodic boundary condition
    
    Parameters
    ----------
    pos: 2D numpy array of floats
        Contains position of all atoms in the system. Axis 0 is the global atom
        ID, and axis 1 is the direction.
    atoms: List of integers
        Contains global ID of atoms in the nanoparticle
    mass: List of floats
        See calc_Rg2_t()

    Returns
    -------
    Rg: float
        Radius of gyration of a group of atoms (nanoparticle) 
    """
    pos_atoms=pos[atoms,:]
    mass_atoms=mass[atoms]
    Mtot=np.sum(mass_atoms)
    com=np.zeros(3)
    for i in range(len(mass_atoms)):
        for j in range(3):
            com[j]+=mass_atoms[i]*pos_atoms[i,j]
    com=com/Mtot
    Rg2=0
    for i in prange(len(mass_atoms)):
        for j in range(3):
            Rg2+=mass_atoms[i]*(pos_atoms[i,j]-com[j])**2
    Rg2=Rg2/Mtot
    
    return Rg2

@njit(parallel=True)
def calc_MOI(pos, atoms, mass):
    """ Calculates moment of intertia tensor for a specific time.
 
    Parameters
    ----------
    pos: 2D numpy array of floats
        Contains position of all atoms in the system. Axis 0 is the global atom
        ID, and axis 1 is the direction.
    atoms: List of integers
        Contains global ID of atoms in the nanoparticle
    mass: numpy array
        See calc_Rg2_t() 

    Returns
    -------
    MOI: 3x3 numpy ndarray of floats
        Moment of interita tensor for a specific time. Units amu.nm2
    """
    pos_atoms=pos[atoms,:]
    mass_atoms=mass[atoms]
    Mtot=np.sum(mass_atoms)
   
    com=np.zeros(3)
    for i in range(len(mass_atoms)):
        for j in range(3):
            com[j]+=mass_atoms[i]*pos_atoms[i,j]
    com=com/Mtot
    I1=np.zeros(3)
    # \sum [ m_i*(x_i-x_com)^2, m_i*(y_i-y_com)^2, m_i*(z_i-z_com)^2 ]
    for i in range(len(mass_atoms)):
        for j in range(3):
            I1[j]+=mass_atoms[i]*(pos_atoms[i,j]-com[j])**2
    I1=I1

    I2=np.zeros(3)
    # \sum [ m_i*(x_i-x_com)(y_i-y_com), m_i*(y_i-y_com)(z_i-z_com),
    #        m_i*(z_i-z_com)(x_i-x_com) ]
    for i in range(len(mass_atoms)):
        for j in range(3):
            I2[j]+=mass_atoms[i]*(pos_atoms[i,j]-com[j])* \
                  (pos_atoms[i,(j+1)%3]-com[(j+1)%3])
    I2=I2

    MOI=np.zeros((3,3))
    for i in range(3):
        MOI[i,i]=I1[(i+1)%3]+I1[(i+2)%3] #Ixx, Iyy or Izz
        MOI[i,(i+1)%3]=-I2[i] #Ixy, Iyz, or Izx
        MOI[(i+1)%3,i]=-I2[i] #Iyz, Izy or Ixz
    return MOI

@njit(parallel=True)
def calc_GyrT(pos, atoms):
    """ Calculates gyration tensor for a specific time.
 
    Parameters
    ----------
    pos: 2D numpy array of floats
        Contains position of all atoms in the system. Axis 0 is the global atom
        ID, and axis 1 is the direction.
    atoms: List of integers
        Contains global ID of atoms in the nanoparticle
    Returns
    -------
    GyrT: 3x3 numpy ndarray of floats
        Gyration tensor for a specific time
    """
    pos_atoms=pos[atoms,:]
    N=len(atoms)
    cen=np.zeros(3)
    for i in range(N):
        for j in range(3):
            cen[j]+=pos_atoms[i,j]
    cen=cen/float(N)
    I1=np.zeros(3)
    # \sum [ (x_i-x_com)^2/N, (y_i-y_com)^2/N, (z_i-z_com)^2/N ]
    for i in range(N):
        for j in range(3):
            I1[j]+=(pos_atoms[i,j]-cen[j])**2
    I1=I1/float(N)

    I2=np.zeros(3)
    # \sum [ (x_i-x_com)(y_i-y_com)/N, (y_i-y_com)(z_i-z_com)/N,
    #        (z_i-z_com)(x_i-x_com)/N ]
    for i in range(N):
        for j in range(3):
            I2[j]+=(pos_atoms[i,j]-cen[j])*(pos_atoms[i,(j+1)%3]-cen[(j+1)%3])
    I2=I2/float(N)

    GyrT=np.zeros((3,3))
    for i in range(3):
        GyrT[i,i]=I1[i] #Ixx, Iyy or Izz
        GyrT[i,(i+1)%3]=I2[i] #Ixy, Iyz, or Izx
        GyrT[(i+1)%3,i]=I2[i] #Iyz, Izy or Ixz
    return GyrT

def NP_shape2(pos_atoms):
    """ Calculates NP shape descriptors asphericitym, acylindricity, and 
        relative shape anisotropy for a NP.
    
    The descriptors are summarized in: 
    https://en.wikipedia.org/wiki/Gyration_tensor
    published article DOI:https://doi.org/10.1021/ma00148a028

    Parameters
    ----------
    pos_atoms: numpyy array of floats
        Array of particle positions.

    Returns
    -------
    b: float
        Asphericity
    c: float
        Acylindricity
    k2: float
        Relative shape anisotropy
    """
    GyrT=calc_GyrT(pos_atoms, np.arange(len(pos_atoms)))
    eig,eigv=np.linalg.eig(GyrT)
    eig=sorted(eig)
    #Quick reference: https://en.wikipedia.org/wiki/Gyration_tensor
    #Please look into detailed references and cite them (not the wiki)
    b=eig[2]-0.5*(eig[0]+eig[1]) #asphericity
    c=eig[1]-eig[0] #acylindricity 
    k2=1.5*np.sum(np.square(eig))/(sum(eig)**2) -0.5 #relative shape anisotropy
    return b, c, k2

def NP_shape(cluster, inGRO='New.gro', ndx_pickle='molndx.pickle', prefix=''):
    """ Calculates NP shape descriptors asphericitym, acylindricity, and 
        relative shape anisotropy for a NP.
    
    The descriptors are summarized in: 
    https://en.wikipedia.org/wiki/Gyration_tensor
    published article DOI:https://doi.org/10.1021/ma00148a028

    Parameters
    ----------
    cluster: 2D list
         Contains DNA and PEI IDs of one cluster.
    inGRO: str, Optional
        File name of input Gromacs file. (Default 'New.gro')
    ndx_pickle: str, Optional
        Filename of the pickled Gromacs index file. See gmx.gen_index_mol() for
        more details. (Default 'molndx.pickle')
    prefix: str
        Prefix for DNA and PEI name. (Default '')

    Returns
    -------
    See NP_shape2()
    """
    constants=nx.read_gpickle('constants.pickle')
    pname=prefix+constants['pei_name']
    dname=prefix+constants['dna_name']
    ndx=nx.read_gpickle(ndx_pickle)
    catoms=[]
    for d in cluster[0]:
        catoms+=list(ndx[dname+str(d)])
    for p in cluster[1]:
        catoms+=list(ndx[pname+str(p)])

    pos,box,text=gmx.read_gro(inGRO)
    b,c,k2=NP_shape2(pos[catoms])
    print('asphericity: '+str(round(b,4)))
    print('acylindricity: '+str(round(c,4)))
    print('relative shape anisotropy: '+str(round(k2,4)))
    return b, c, k2

def NP_shape_all(shape_pickle='shape.pickle', inGRO='New', sep=' ', \
    cluster_pickle='cluster.pickle', main_mol=0, ndx_pickle='molndx.pickle', \
    prefix=''):
    """ Calculates NP shape descriptors asphericitym, acylindricity, and 
        relative shape anisotropy for all NP at all time.
    
    The descriptors are summarized in: 
    https://en.wikipedia.org/wiki/Gyration_tensor
    published article DOI:https://doi.org/10.1021/ma00148a028

    Parameters
    ----------
    shape_pickle: str, Optional
        Filename in which shape discriptors will be saved as pickled file. 
        (Default 'shape.pickle')
    inGRO: str, Optional
        Starting file name of input Gromacs files. Files [infile][t].gro are
        read. The nanoparticles must be whole (across boundary). 
        (Default 'New')
    sep: str, Optional
        A string that separates data. (Default ' ')
    cluster_pickle: str, Optional
        Filename which contains pickled cluster data. See cluster.gen_cluster()
        for more details. (Default 'cluster.pickle')
    main_mol: int, Optional
        Chooses a main molecule to calculate size of nanoparticle. 0 represents
        DNA and 1 represents PEI. (Default 0).
    ndx_pickle: str, Optional
        Filename of the pickled Gromacs index file. See gmx.gen_index_mol() for
        more details. (Default 'molndx.pickle')
    prefix: str
        Prefix for DNA and PEI name. (Default '')
        
    Writes
    ------
    [shape_pickle]:
        Axis 0 is timestep, Axis 1 is cluster ID, Axis 2 stores [0] aspericity
        [1] acylindirciy, and [2] relative shape anisotropy
    """

    clusters=nx.read_gpickle(cluster_pickle)
    constants=nx.read_gpickle('constants.pickle')
    ndx=nx.read_gpickle(ndx_pickle)
    pname=prefix+constants['pei_name']
    dname=prefix+constants['dna_name']
    times=len(clusters)
    shape=[]
    for t in range(times):
        pos,box,text=gmx.read_gro(inGRO+str(t)+'.gro')
        catoms=gmx.get_NPatomIDs(cluster[t],ndx,dna_name,pei_name,main_mol)
        shape_c=[]
        for x in len(catoms):
            b,c,k2=NP_shape2(pos[catoms[x]])
            shape_c.append([b, c, k2])
        shape.appen(shape_c)
    print('Writing: '+shape_pickle)
    nx.write_gpickle(shape,shape_pickle)

def calc_Rh_Rg(Rh_pickle='Rh.pickle', Rg2_pickle='Rg.pickle', inGRO='New',\
    mass_pickle='mass.pickle', cluster_pickle='cluster.pickle', main_mol = 0, \
    ndx_pickle='molndx.pickle', sep=' ',prefix=''):
    """Calculates hydrodynamic radius (Rh) and square of radius of gyration 
       (Rg)

    Parameters
    ----------
    Rh_pickle: str, Optional
        Filename to pickle hydrodynamic radius data. (Default 'Rh.pickle')
    Rg2_pickle: str, Optional
        Filename to pickle square of radius of gyraiton data. 
        (Default 'Rg.pickle')
    inGRO: str, Optional
        Starting file name of input Gromacs files. Files [infile][t].gro are
        read. The nanoparticles must be whole across boundary. (Default 'New')
    mass_pickle: str, Optional
        Filename containing pickled mass data. Contains a list of atom mass,
         ordered according to the global ID of atoms. (Default 'mass.pickle')
    cluster_pickle: str, Optional
        Filename which contains pickled cluster data. See cluster.gen_cluster()
        for more details. (Default 'cluster.pickle')
    main_mol: int, Optional
        Chooses a main molecule to calculate size of nanoparticle. 0 represents
        DNA and 1 represents PEI. (Default 0).
    ndx_pickle: str, Optional
        Filename of the pickled Gromacs index file. See gmx.gen_index_mol() for
        more details. (Default 'molndx.pickle')
    sep: str, Optional
        A string that separates data. (Default ' ')
    prefix: str
        Prefix for DNA and PEI name. (Default '')

    Writes
    ------
    [Rh_pickle]: 2D list of floats
        Axis 0 and 1 represents time and cluster ID. Matches the order of 
        pickled cluster data (clusters followed by free main molecule). 
        Contains hydrodynamic radius. 
    [Rg2_pickle]: 2D list of floats
        Axis 0 and 1 represents time and cluster ID. Matches the order of 
        pickled cluster data (clusters followed by free main molecule). 
        Contains square of radius of gyration.
    """

    constants=nx.read_gpickle('constants.pickle')
    ndna=constants['ndna']
    npei=constants['npei']
    dna_name=prefix+constants['dna_name']
    pei_name=prefix+constants['pei_name']
    if main_mol==0:
        main_name=dna_name
    elif main_mol==1:
        main_name=pei_name
    mass=nx.read_gpickle(mass_pickle)
    ndx=nx.read_gpickle(ndx_pickle)
    cluster=nx.read_gpickle(cluster_pickle)
    times=len(cluster)

    if Rh_pickle!=None:
        Rh_data=[]
    if Rg2_pickle!=None:
        Rg2_data=[]
 
    for t in range(times):
        print("Reading: "+inGRO+str(t)+".gro     ",end="\r")
        pos,box,text=gmx.read_gro(inGRO+str(t)+'.gro')
        catoms=gmx.get_NPatomIDs(cluster[t],ndx,dna_name,pei_name,main_mol) #Change function.
        if Rh_pickle!=None:
            Rhs=calc_Rh_t(pos,catoms)
            Rh_data.append(Rhs)
        if Rg2_pickle!=None:
            Rg2s=calc_Rg2_t(pos,catoms,mass)
            Rg2_data.append(Rg2s)
    foo="Writing: "
    if Rh_pickle !=None:
        foo+=Rh_pickle+" "
        nx.write_gpickle(Rh_data,Rh_pickle)
    if Rg2_pickle !=None:
        foo+=Rg2_pickle
        nx.write_gpickle(Rg2_data,Rg2_pickle)
    print("\n"+foo)

def gen_rad_avg(rad_pickle,avg_len,outname,sep=' ',time_pickle='time.pickle', \
    sqrt=False):
    """Calculates the average radius over time
    
    Parameters
    ----------
    rad_pickle: pickled 2D list of floats or integers
        Contains radius values (Ex: Rh, Rg) at different timesteps and for 
        different nanoparticles
    avg_len: int
        Number of timesteps over which the average is evaluated
    outname: str 
        Output filename
    sep: str, Optional
        A string that separates data. (Default ' ')
    time_pickle: str, Optional
        Filename which contains the pickled simulation time data. Contains a
        numpy array of simulation time, which is adjusted by multiplying a 
        factor and shifting by a constant. (Default 'time.pickle')
    sqrt: bool, Optional
        Specifies if a square root should be performed over the average and 
        standard error. Use True for taking a square root for square of radius
        of gyraiton values. (Default False)

    Writes
    ------
    [outname]: 
        First line is a comment. Each subsequent line contains time, average 
        radius, and its standard error.
    """
    print("Writing: "+outname)
    rad=nx.read_gpickle(rad_pickle)
    sim_time=nx.read_gpickle(time_pickle)
    times=len(rad)
    w=open(outname,'w')
    w.write('#time'+sep+'avg_radius'+sep+'std_error\n')
    for t in range(times):
        if t==0:
            R=[]
        R=R+rad[t] 
        if t%avg_len==avg_len-1 or t==times-1:
            avg=np.average(R)
            std=np.std(R)
            if sqrt:
                avg=np.sqrt(avg)
                std=np.sqrt(avg)
            tavg=np.average(sim_time[int(t/avg_len)*avg_len:t+1])
            w.write(str(round(tavg,4))+sep+str(round(avg,4))+sep+ \
                str(round(std,4))+'\n')
    w.close()

def gen_rad_per_size(rad_pickle, outname, t1, t2, sep=' ', main_mol=0, \
    cluster_pickle='cluster.pickle',sqrt=False):
    """Calculates average radius as a function of number average size of 
       nanoparticle

    Parameters
    ----------
    rad_pickle: pickled data
        Contains radius (Rh or Rg) values at different timesteps and for 
        different nanoparticles
    outname: str 
        Output filename
    t1: int
         time step to start averaing (included).
    t2: int
         time step to end averaging (non included).
    sep: str, Optional
        A string that separates data. (Default ' ')
    main_mol: int, Optional
        Chooses a main molecule to calculate size of nanoparticle. 0 represents
        DNA and 1 represents PEI. (Default 0).
    cluster_pickle: str, Optional
        Filename which contains pickled cluster data. See cluster.gen_cluster()
        for more details. (Default 'cluster.pickle')
    sqrt: bool, Optional
        Specifies if a square root should be performed over the average and 
        standard error. Use True for taking a square root for square of radius
        of gyraiton values. (Default False)
   
    Writes
    ------
    [outname]: 
        First line is a comment. Each subsequent line has the number average 
        size of the nanoparticle, average radius and its standard error.
    """
    print("Writing: "+outname)
    constants=nx.read_gpickle('constants.pickle')
    ndna=constants['ndna']
    npei=constants['npei']
    dna_name=constants['dna_name']
    pei_name=constants['pei_name']
    
    rad=nx.read_gpickle(rad_pickle)
    cluster=nx.read_gpickle(cluster_pickle)
    times=len(rad)
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
     
    R=np.zeros(N)
    R2=np.zeros(N)
    cnt=np.zeros(N)
    for t in range(t1,t2):
        Nclust=len(cluster[t])-2
        for cid in range(Nclust):
            size=len(cluster[t][cid][main_mol])
            R[size-1]+=rad[t][cid]
            R2[size-1]+=rad[t][cid]**2
            cnt[size-1]+=1
        for cid in range(Nclust,len(rad[t])):
            R[0]+=rad[t][cid]
            R2[0]+=rad[t][cid]**2
            cnt[0]+=1
    w.write('#Size'+sep+'avg_radius'+sep+'std err\n')
    for i in range(N):
        if cnt[i]==0:
            avg=0
            std=0
        else:
            avg=R[i]/cnt[i]
            std=np.sqrt(R2[i]/cnt[i]-avg**2)
            if sqrt:
                avg=np.sqrt(avg)
                std=np.sqrt(std)
        w.write(str(i+1)+sep+str(round(avg,4))+sep+str(round(std,4))+'\n')
    w.close()
