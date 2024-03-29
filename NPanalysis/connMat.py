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
import numpy as np
import networkx as nx
from . import gmx
from . import geometry
small=1E-16

def gro2connected(inGRO='DP',connected_pickle='connected.pickle',
    ndx_pickle='molndx.pickle', time_shift=0, time_fac=1, \
    time_pickle='time.pickle', mindist_pickle='mindist.pickle',
    prefix=''):
    """Convert .gro files to connection matrix and change required to make 
       molecules whole.

    Parameter
    ---------
    inGRO: str, Optional
        The starting strings of input Gromacs structure files. Files 
        [inGRO][t].gro are read, where [t] is the simulation time. Molecules 
        must be whole in the .gro files.These can be generated using 
        'gmx trjconv -pbc whole -sep' (Default 'DP')
    connected_pickle: str, Optional
        Filename to save pickled connection matrix data. 
        (Default 'connected.pickle')
    ndx_pickle: str, Optional
        Filename of the pickled Gromacs index file. See gmx.gen_index_mol() for 
        more details. (Default 'molndx.pickle')
    time_shift: float, Optional
        Shift the simulation time by a constant (after scaling. (Default 0)
    time_fac: float, Optional
        Multiply the simulation time by a factor. (Default 1)
    time_pickle: str, Optional
        Filename to save the pickled simulation time. (Default 'time.pickle')
    mindist_pickle: str, Optional
        Filename of save pickled minimum distance data. 
        (Default 'mindist.pickle')
    prefix: str, Optional
        Prefix for DNA and PEI name. (Default '')

    Writes
    ------
    [connected_pickle]: pickled 3D numpy ndarry of bool. 
        Contains pickled connection matrix. Axis 0, 1, and 2 corressponds to 
        timestep, DNA ID, and PEI ID respectively. If the DNA-PEI pair is 
        connected the value is True.    
    [time_pickle]: pickled 1D numpy array of float.
        Contains pickled simulation time. Axis 0 is the timestep. The simulation
        time is scaled by a factor and shifted.
    [mindist_pickle]: pickled 3D numpy ndarray of float.
        Contains pickled minimum distance data. Axis 0, 1, and 2 corresponds to
        timestep, DNA ID, and PEI ID respectively. The value of the array is the
        minimum distance between the DNA-PEI pair. 
    """
    constants=nx.read_gpickle('constants.pickle')
    ndna=constants['ndna']
    npei=constants['npei']
    pbc=np.array(constants['pbc'])
    dna_name=prefix+constants['dna_name']
    pei_name=prefix+constants['pei_name']
    contact_dist=constants['contact_dist']
    ndx=nx.read_gpickle(ndx_pickle)
    adna=len(ndx[dna_name+'0'])
    apei=len(ndx[pei_name+'0']) 
    #Determine number of timesteps.
    times=0
    while os.path.exists(inGRO+str(times)+'.gro'):
        times+=1

    #####INITIALIZE##########
    #connected: connection matrix for all timesteps
    connected=np.zeros((times,ndna,npei),dtype=bool)
    #mindist: Minimum distance (with PBC) between DNA and PEI for all timesteps
    mindist=np.zeros((times,ndna,npei))
    sim_time=[]
    time_method='sim'
    for t in range(times):
        print("Reading: "+inGRO+str(t)+'.gro            ',end="\r")
        pos,box,texts=gmx.read_gro(inGRO+str(t)+'.gro')
        foo=texts[0].split('t=')#
        foo=foo[1].split()
        if t==0:
            try:
                sim_time.append(float(foo[0])*time_fac+time_shift)
            except: 
                print("\nGRO files does not have time in it. Using timesteps instead")
                time_method='step'
                sim_time.append(t)
        else:
            if time_method=='sim':
                sim_time.append(float(foo[0])*time_fac+time_shift)
            elif time_method=='step':
                som_time.append(t)
        
        d_atoms=[] # Global atom ID of all DNA molecules
        for d in range(ndna):
            d_atoms+=ndx[dna_name+str(d)]
        d_atoms=np.array(d_atoms)
        p_atoms=[] # Global atom ID of all PEI molecules
        for p in range(npei):
            p_atoms+=ndx[pei_name+str(p)]
        p_atoms=np.array(p_atoms)
  
        # dmin: Minimum distance (with PBC) between all DNA-PEI pair 
        # disp_t: Displacement of DNA requried to minimize distance between 
        #         DNA-PEI pair at timestep t.
        dmin,disp_t=geometry.get_pbc_mindist(pos[d_atoms,:],pos[p_atoms,:], \
            box,ndna,npei,adna,apei,pbc)
        mindist[t,:,:]=dmin[:,:]
        for d in range(ndna):
            for p in range(npei):
                if dmin[d,p]<contact_dist:
                    connected[t,d,p]=True

    print("Writing: "+connected_pickle+" "+time_pickle+" "+mindist_pickle)
    nx.write_gpickle(connected,connected_pickle)
    nx.write_gpickle(sim_time,time_pickle)
    nx.write_gpickle(mindist,mindist_pickle)
    
def mindist2connected(connected_pickle='connected.pickle', \
    mindist_pickle='mindist.pickle'): 
    """Reads pickled minimum distances between DNA-PEI pairs and calculates the
       the connection matrix of DNAs and PEIs. 

    Parameters
    ----------
    connected_pickle: str, Optional
        See gro2connected() (Default 'connected.pickle')
    mindist_pickle: str, Optional
        See gro2connected(). (Default 'mindist.pickle')
    
    Writes
    ------
    [connected_pickle]: pickled 3D numpy ndarray of booleans.
        For details see connMat.gro2connected() 
    """
    print("Reading: "+mindist_pickle)
    mindists=nx.read_gpickle(mindist_pickle)
    #Read constants.
    times,ndna,npei=mindists.shape
    constants=nx.read_gpickle('constants.pickle')
    contact_dist=constants['contact_dist']
    connected=mindists<contact_dist
    #Pickle Data
    print("Writing: "+connected_pickle)
    nx.write_gpickle(connected,connected_pickle)

def write_connMat(connected_pickle='connected.pickle', outheader='connected'): 
    """Writes connection matrix in readable format

    Parameters
    ----------
    connected_pickle: str, Optional
        See gro2connected(). (Default 'connected.pickle')
    outheader: str, Optional
        Starting strings of output file name. (Default 'connected')

    Writes
    ------
    [outheader][t].dat: txt file format
        Writes ndna X npei digits. 0 and 1 represents the dna-pei pair are 
        non-connected and connected. Row represents dnas and column represents
        peis.
    """
    print("Writing: "+outheader+"[t].dat")
    connected=nx.read_gpickle(connected_pickle)
    times=len(connected)
    const=connected.shape
    for t in range(times):
        w=open(outheader+str(t)+'.dat','w')
        for d in range(const[1]):
            for p in range(const[2]):
                #Note: There is no separator between digits 0, 1. 
                w.write(str(int(connected[t,d,p])))
            w.write('\n')
        w.close()
    
 
def get_roles(avg_step, connected_pickle='connected.pickle', main_mol = 1, 
    time_pickle='time.pickle', outname='PEI_roles.dat', sep=" "): 
    """Calculates the average number of molecules (PEI or DNA) in each role:
       free, peripheral, and bridging as a function of time.  

    Parameters
    ----------
    avg_step: int
        Number of timesteps over which the average is evaluated
    connected_pickle: str, Optional
        See gro2connected(). (Default 'connected.pickle')
    main_mol: int, Optional
        Decides the molecule for which roles, free, peripheral, and bridging are
        calculated. 0 is for DNA, and 1 is for PEI. (Default 1)
    time_pickle: str, Optional
        See gro2connected(). (Default 'time.pickle')
    outname: str, Optional
        Output file name. (Default 'PEI_roles.dat')
    sep: str, Optional
        See __init__.cat_pickle(). (Default ' ') 

    Writes
    ------
    [outname]: txt file format
        First line is a comment. Each subsequent line contains time, average
        number of free PEIs, peripheral PEIs and bridging PEIs.  
    """
    print("Writing: "+outname)
    constants=nx.read_gpickle('constants.pickle')
    dna_name=constants['dna_name']
    pei_name=constants['pei_name']

    sim_time=nx.read_gpickle(time_pickle)
    connected=nx.read_gpickle(connected_pickle)
    times=len(connected)
    w=open(outname,"w")
    # Axis 1 represents DNA IDs. Summing connected along axis 1 results in a 2D
    # numpy array, where axis 0 is time and axis 1 is PEI IDs. The value of 
    # roles is the number of DNAs connected to a PEI ID at a given time.
    #
    # To get number of PEIs connected to a DNA ID at a given time, connected
    # should be summed over axis 2.
    if main_mol==1: #PEI
        roles=np.sum(connected,axis=1) #sum over DNA axis
        w.write("# Bridging molecule is "+pei_name+'\n')
    elif main_mol==0: #DNA
        roles=np.sum(connected,axis=2) #sum over PEI axis
        w.write("# Bridging molecule is "+dna_name+'\n')

    # Free molecules: PEIs not bound to any DNAs (main_mol=1) or DNAs not bound to
    #                 any PEIs (main_mol=0) 
    free=np.sum((roles==0),axis=1)
    # Peripheral molecules: PEIs connected to exactly one DNA (main_mol=1) or
    #                       DNAs connected to exactly one PEI (main_mol=0)
    peri=np.sum((roles==1),axis=1)
    # Bridging molecules: PEIs connected to more than one DNA (main_mol=1) or
    #                     DNAs connected to more than one PEI (main_mol=0)
    bri=np.sum((roles>1),axis=1)
     
    w.write('#time'+sep+'number_of_free'+sep+'number_of_peripheral'+ \
        sep+'number_of_bridging\n')
    if avg_step==0:
        for t in range(times):
            w.write(str(round(sim_time[t],4)) + sep + str(free[t]) + sep + 
                str(peri[t]) + sep + str(bri[t]) + '\n')
    else:
        for t in range(int(times/avg_step)):
            t1=t*avg_step
            t2=(t+1)*avg_step
            tavg=np.average(sim_time[t1:min(t2+1,times)])
            w.write( str(round(tavg,4)) + sep + \
               str(round(np.average(free[t1:t2+1]),4)) + sep + \
               str(round(np.average(peri[t1:t2+1]),4)) + sep + \
               str(round(np.average(bri[t1:t2+1]),4)) + '\n')
    w.close()

def get_roles2(avg_step, connected_pickle='connected.pickle', main_mol = 0, 
    time_pickle='time.pickle', outname='PEI_roles2.dat', sep=" "):
    """Calculates the average number of bridging molecules between bridged 
       molecule pairs and number of bridged molecule pairs as a function
       of time. 
  
    main_mol=0 => average number of bridging PEI between bridged DNA pair,
                  number of bridged DNA pairs.
    main_mol=1 => average number of bridging DNA between bridged PEI pair, 
                  number of bridged PEI pairs.

    Parameters
    ----------
    See get_roles() 

    Writes
    ------
    [outname]: txt file format
        First line is a comment. Each subsequent line contains time, average 
        number of bridging PEI between bridged DNA pairs, and the total number
        of bridged DNA pairs.  
    """
    print("Writing: "+outname)
    constants=nx.read_gpickle('constants.pickle')
    dna_name=constants['dna_name']
    pei_name=constants['pei_name']

    connected=nx.read_gpickle(connected_pickle)
    sim_time=nx.read_gpickle(time_pickle)
    const=connected.shape
    w=open(outname,"w")

    if main_mol==1: #PEI
        w.write("# Bridging molecule is "+pei_name+'\n')
    elif main_mol==0: #DNA
        w.write("# Bridging molecule is "+dna_name+'\n')

    w.write('#time'+sep+'average_number_of_bridges_between_pairs'+sep+ \
        'number_of_bridged_pairs\n')
    for t in range(const[0]):
        if avg_step==0:
            avg_bri=0
            m2m=0 
        elif t%avg_step==0: #Initialize average to value to zero 
            avg_bri=0 #Average number of bridges between molecule pairs
            m2m=0 #Number of molecule pairs
        #Itereate over molecule pairs. Avoid double counting.
        for m1 in range(const[2-main_mol]-1):
            for m2 in range(m1+1,const[2-main_mol]):
                if main_mol==1:
                    #Number of bridgeing PEI between a DNA pair.
                    bri=np.sum(np.multiply(connected[t,m1,:],connected[t,m2,:]))
                elif main_mol==0:
                    #Number of bridgeing DNA between a PEI pair.
                    bri=np.sum(np.multiply(connected[t,:,m1],connected[t,:,m2]))
                if bri>0:
                    avg_bri+=bri
                    m2m+=1
        if avg_step==0:
            w.write(str(round(sim_time[t],4)) + sep + str(round(avg_bri/(m2m+small),4)) +
                    sep + str(round(m2m,4)) + '\n')
        elif t%avg_step==avg_step-1:
            avg_bri/=float(m2m+small) #Average over molecule pairs
            m2m/=float(avg_step) #Average over time
            tavg=np.average(sim_time[int(t/avg_step)*avg_step:t+2])
            w.write(str(round(tavg,4)) + sep + \
                str(round(avg_bri,4)) + sep + str(round(m2m,4)) + '\n')
    w.close()  
 
def get_role_conversion(connected_pickle='connected.pickle', main_mol=1, \
    outname='role_conv.dat',sep=' '):
    """ Calculates total number of conversions between each rol (unbound,
        peripheral, and bridging) in consecutive timesteps.

    Parameters
    ----------
    See get_roles()

    Writes
    ------
    [outname]: txt file format
        First line is a comment. Second line contains role conversions, unbound
        to peripheral, peripheral to bridging, and unbound to bridging. Third
        line is a commend. Fourth line contains role conversions, peripheral to
        unbound, bridging to peripheral, and bridging to unbound.
    """
    connected=nx.read_gpickle(connected_pickle)
    consts=connected.shape
    role_conv=np.zeros((3,3),dtype=int)
    # 0,1,2 implies free, peripheral, and bridging
    # role_conv[i,j] implies conversion from i-> j
    # conversions i->i (diagonal terms) should be ignored. 
    roles=np.sum(connected,axis=2-main_mol)
    constants=nx.read_gpickle('constants.pickle')
    dna_name=constants['dna_name']
    pei_name=constants['pei_name']


    for t in range(consts[0]):
        for m in range(consts[1+main_mol]):
            if roles[t,m]>1:
                roles[t,m]=2
    for t in range(consts[0]-1):
        for m in range(consts[1+main_mol]):
            if roles[t+1,m]!=roles[t,m]:
                role_conv[roles[t,m],roles[t+1,m]]+=1
    w=open(outname,'w')
    if main_mol==1: #PEI
        w.write("# Bridging molecule is "+pei_name+'\n')
    elif main_mol==0: #DNA
        w.write("# Bridging molecule is "+dna_name+'\n')
    w.write('#u->p\tp->b\tu->b\n')
    w.write(str(role_conv[0,1])+sep+str(role_conv[1,2])+sep+\
            str(role_conv[0,2])+'\n')
    w.write('#p->u\tb->p\tb->u\n')
    w.write(str(role_conv[1,0])+sep+str(role_conv[2,1])+sep+\
            str(role_conv[2,0])+'\n')
    w.close()
