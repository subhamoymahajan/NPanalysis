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
from numba import njit, prange, int64
import numpy as np

def distance(pos_i,pos_j):
    """Returns distance between two position coordinate. 

    Does not consider periodic boundary condition
    
    Parameters:
    ----------
    pos_i: numpy array of floats
        Position coordinates of atom i
    pos_j: numpy array of floats
        Position coordinates of atom j
    
    Returns
    -------
    d2: float
        Distance between two atoms
    """
    return np.sqrt(distance2(pos_i,pos_j))

def distance2(pos_i,pos_j):
    """Returns square of distance between two position coordinate. 

    Does not consider periodic boundary condition
    
    Parameters:
    ----------
    pos_i: numpy array of floats
        Position coordinates of atom i
    pos_j: numpy array of floats
        Position coordinates of atom j
    
    Returns
    -------
    d2: float
        Square of distance between two atoms
    """
    return np.sum(np.square(pos_i-pos_j)) 

@njit(parallel=True)
def dist_matrix(pos,maxd=None):
    """Calculates distance matrix for positions
    """
    N=len(pos)
    dist_mat=np.zeros((N,N))
    if maxd!=None:
        maxd2=maxd*maxd
    for i in prange(N-1):
        for j in prange(i+1,N):
            foo=np.sum(np.square(pos[i]-pos[j]))
            if maxd==None:
                dist_mat[i,j]=np.sqrt(foo)
            elif foo>maxd2:
                dist_mat[i,j]=maxd
            else:
                dist_mat[i,j]=np.sqrt(foo)
    return dist_mat       
            
   

@njit(parallel=True)
def get_cen(pos, atoms):
    """Calculates geometric center of a group of atoms without PBC boundary
       conditions
 
    Parameters
    ----------
    pos: 2D numpy array of floats
       Contains position of all atoms in the system. Axis 0 is the global atom
       ID, and axis 1 is the direction.
    atoms: numpy array of integers
       numpy array of global IDs of atoms.
    Returns
    -------
    cen: numpy array of floats
        Geometric center of group of atoms
    """
    pos_atoms=pos[atoms,:]
    cen=np.sum(pos_atoms,axis=0)/len(pos_atoms)
    return cen  

@njit(parallel=True)  
def get_pbc_mindist(dna_pos,pei_pos,box,ndna,npei,adna,apei, pbc): 
    """Calculate the amount of boxlengths (in x, y, and z direction) required 
       to move group of atoms (atoms2) to minimize distance between groups of
       atoms (atoms2 and atoms1).

    Parameters
    ----------
    box: numpy array of floats
       Dimensions of cuboidal simulation box in x, y and z. 
    pos1: 2D array of floats
       Positions of a group of atoms (1)
    pos2: list of int
       Positions of a group of atoms (2)
    pbc: list of integer
       List specifying if periodic boundary condition (PBC) is applied in 
       x, y, and z directions.

    Returns
    -------
    dmin: 2D numpy array of float
        Minimum distance considering PBC. Axis 0 and 1 represent DNA and PEI ID
    mol_disp: 2D numpy array of float 
        Displacement of DNAs needed to minimize distance with PEIs. 
        Axis 0 and 1 represents DNA ID and PEI ID respectively.
    """
    # Definitions:
    # change: The amount of box lengths in x, y, and z directions required to 
    #         move DNA molecules to minimize distance between DNA and PEI 
    #         molecules
    # PBC distance: distance calculated by apply periodic boundary condition 
    #               based on the pbc variable. pbc[k] = 1 implies pbc is active
    #               in direction k, 0 implies it is inactive. k = 0, 1, 2 
    #               implies directions x, y, and z respectively.
    N1=len(dna_pos)
    N2=len(pei_pos)
    # Position of DNA atoms relative to PEI atoms
    dr=np.zeros((N1,N2,3))
    dr2=np.zeros((N1,N2,3))
    #displacement of DNA needed to minimize distance with PEI
    #disp=np.zeros((N1,N2,3)) 
    for i in prange(N1):
        for j in prange(N2):
            for k in range(3):
                dr[i,j,k]=dna_pos[i][k]-pei_pos[j][k] #relative position
    
    for k in range(3):
        if pbc[k]==1:#Apply PBC 
            dr2[:,:,k]=np.mod(dr[:,:,k]+0.5*box[k],box[k])-0.5*box[k] 
        else:
            dr2[:,:,k]=dr[:,:,k]
    # Displacement of DNA atoms needed to minimize distance with PEI atoms
    disp=dr2-dr 
    d2=np.sum(np.square(dr2),axis=2) #square distance with PBC 
    dmin2=np.zeros((ndna,npei)) #Intialize minimum distance between DNA-PEI
    # Displacement of DNA molecules needed to minimize distance with PEI.
    mol_disp=np.zeros((ndna,npei,3)) 
    N=np.zeros((ndna,npei),dtype=int64)
    for d in prange(ndna):
        for p in prange(npei):
            N[d,p]=np.argmin(d2[d*adna:(d+1)*adna,p*apei:(p+1)*apei])
            dmin2[d,p]=d2[d*adna+int(N[d,p]/apei),p*apei+np.mod(N[d,p],apei)]
            mol_disp[d,p,:]=disp[d*adna+int(N[d,p]/apei), \
                p*apei+np.mod(N[d,p],apei),:]
    dmin=np.sqrt(dmin2) #taking sqrt outside the for loop reduces computation.
    return dmin,mol_disp

@njit(parallel=True)
def get_pbcdisp_mols(pos, box, atoms1, atoms2, pbc): #@
    """Calculate the displacement required to move atoms2 to minimize distance
        between groups of atoms2 and atoms1

    Parameters
    ----------
    pos: 2D numpy array of floats
       Contains position of all atoms in the system. Axis 0 is the global atom
       ID, and axis 1 is the direction.
    box: numpy array of floats
       Dimensions of cuboidal simulation box in x, y and z. 
    atoms1: list of int
       Global ID of atoms in a group (1)
    atoms2: list of int
       Global ID of atoms in a group (2)
    pbc: list of integer
       List specifying if periodic boundary condition (PBC) is applied in 
       x, y, and z directions.

    Returns
    -------
    min_disp: array of ints
        Contains the displacement of atoms1 needed to minimize its distance with
         atoms2 
    """
    # Definitions:
    # change: The amount of box lengths in x, y, and z directions required to 
    #         move group of atoms2 to minimize distance between atoms2 and
    #         atoms1
    # PBC distance: distance calculated by apply periodic boundary condition 
    #               based on the pbc variable. pbc[k] = 1 implies pbc is active
    #               in direction k, 0 implies it is inactive. k = 0, 1, 2 
    #               implies directions x, y, and z respectively. 
    dr=np.zeros((len(atoms1),len(atoms2),3))
    dr2=np.zeros((len(atoms1),len(atoms2),3))
    for i in prange(len(atoms1)):
        for j in prange(len(atoms2)):
            # Change required to minimize PBC distance between a2 and a1
            for k in range(3):
                dr[i,j,k]=pos[atoms2[j]][k]-pos[atoms1[i]][k] #relative distance in direction k.
    
    for k in range(3):
        if pbc[k]==1:#Apply PBC 
            dr2[:,:,k]=np.mod(dr[:,:,k]+0.5*box[k],box[k])-0.5*box[k] 
        else:
            dr2[:,:,k]=dr[:,:,k]
    disp=dr2-dr
    d2=np.sum(np.square(disp),axis=2)
    N=np.argmin(d2)
    i=int(N/len(d2[0]))
    j=N-len(d2[0])*i
    min_disp=disp[i,j,:]
    return min_disp

@njit
def change_pos(pos,disp,atoms): 
    """Change position of a DNA or PEI molecule
    
    Parameters
    ----------
    pos: 2D numpy array of floats
       Contains position of all atoms in the system. Axis 0 is the global atom
       ID, and axis 1 is the direction.
    disp: numpy array of integers
       Displacement to be applied to atoms.
    atoms: numpy array of integers
       List of global IDs of atoms. 
  
    Returns
    -------
    pos: 2d numpy array of floats
       Updated position of atoms in the atoms listtem. Contains position of all
       atoms in the system. Axis 0 is the global atom ID, and axis 1 is the 
       direction.
    """
    pos[atoms,:]+=disp
    return pos  
 
@njit(parallel=True)
def change_NPpos(pos, box, atoms, pbc): 
    """Change positions of a NP using PBC to move the geometric center of the 
       molecule in the primary simulation box.
   
    Parameters
    ----------
    pos: 2D numpy array of floats
       Contains position of all atoms in the system. Axis 0 is the global atom
       ID, and axis 1 is the direction.
    box: numpy array of floats
       Dimensions of cuboidal simulation box in x, y and z. 
    atoms: List of integers
       List of global IDs of atoms
    pbc: list of integer
       List specifying if periodic boundary condition (PBC) is applied in 
       x, y, and z directions.

    Returns
    -------
    pos: 2d numpy array of floats
       Updated position of atoms to move the geometric of the nanoparticle 
       within the simulation box. Contains position of all atoms in the system.
       Axis 0 is the global atom ID, and axis 1 is the direction.
    """
    pos_atoms=pos[atoms,:]
    np_cen=np.sum(pos_atoms,axis=0)
    np_cen=np_cen/len(atoms)
    dr=np.mod(np_cen,box)-np_cen
    pos[atoms,:]+=dr
    return pos

