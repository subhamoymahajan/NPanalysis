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
    d: float
        Distance between two atoms
    """
    return np.sqrt(distance2(pos_i,pos_j))

def distance2(pos_i,pos_j):
    """Returns square of distance between two position coordinate. 

    Does not consider periodic boundary condition
    
    Parameters:
    ----------
    See distance()
 
    Returns
    -------
    d2: float
        Square of distance between two atoms
    """
    return np.sum(np.square(pos_i-pos_j)) 

@njit(parallel=True)
def dist_matrix(pos,maxd=None):
    """Calculates distance matrix for positions

    Parameters
    ----------
    pos: 2D numpy array of floats
        See pos_i or pos_j in distance()
    maxd: float
        If square distances are above maxd^2, distance is set as maxd. This is 
        to avoid costly square root evaluations. Default is None, which 
        calculates every square root.

    Returns
    -------
    dist_mat: 2D numpy array of floats
        Distances are calculated for j>i. Distances are cutoff at maxd.
    """
    N=len(pos)
    dist_mat=np.zeros((N,N))
    if maxd!=None:
        maxd2=maxd*maxd
    else:
        maxd2=np.inf
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
        See pos_i or pos_j in distance()
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
def get_pbc_mindist(pos_i,pos_j,box,npos_i,npos_j,apos_i,apos_j, pbc): 
    """Calculate the amount of boxlengths (in x, y, and z direction) required 
       to move group of molecules in (pos_j) to minimize distance between groups
       of molecules present in (pos_i and pos_j).

    Parameters
    ----------
    pos_i: 2D numpy array of floats
       see distance()
    pos_j: 2D numpy array of floats
       see distance()
    box: numpy array of floats
       Dimensions of cuboidal simulation box in x, y and z. 
    npos_i: int
       Number of molecules in pos_i
    npos_j: int
       Number of molecules in pos_k
    apos_i: int
       Number of particles (or atoms) per molecule in pos_i
    apos_j: int
       Number of particles (or atoms) per molecule in pos_j
    pbc: list of integer
       List specifying if periodic boundary condition (PBC) is applied in 
       x, y, and z directions.

    Returns
    -------
    dmin: 2D numpy array of float
        Minimum distance considering PBC. Axis 0 and 1 represent pos_i and 
        pos_j molecules IDs.
    mol_disp: 2D numpy array of float 
        Displacement of pos_i needed to minimize distance with pos_j. 
        Axis 0 and 1 represents pos_i and pos_j molecule IDs respectively.
    """
    # Definitions:
    # PBC distance: distance calculated by apply periodic boundary condition 
    #               based on the pbc variable. pbc[k] = 1 implies pbc is active
    #               in direction k, 0 implies it is inactive. k = 0, 1, 2 
    #               implies directions x, y, and z respectively.
    N1=len(pos_i)
    N2=len(pos_j)
    # Position of pos_j atoms relative to pos_i atoms
    dr=np.zeros((N1,N2,3)) #displacement pos_i-pos_j
    dr2=np.zeros((N1,N2,3)) #displacement pos_i-pos_j after PBC
    
    #disp=np.zeros((N1,N2,3)) 
    for i in prange(N1):
        for j in prange(N2):
            for k in range(3):
                dr[i,j,k]=pos_i[i][k]-pos_j[j][k] #relative position
    for k in range(3):
        if pbc[k]==1:#Apply PBC 
            dr2[:,:,k]=np.mod(dr[:,:,k]+0.5*box[k],box[k])-0.5*box[k] 
        else:
            dr2[:,:,k]=dr[:,:,k]
    # Displacement of move pos_j atoms to minimize distance with pos_i atoms
    disp=dr2-dr
    d2=np.sum(np.square(dr2),axis=2) #square distance with PBC 
    dmin2=np.zeros((npos_i,npos_j)) #Intialize minimum distance 
    # Displacement of pos_j molecules needed to minimize distance with 
    # pos_i molecules.
    mol_disp=np.zeros((npos_i,npos_j,3)) 
    N=np.zeros((npos_i,npos_j),dtype=int64)
    for d in prange(npos_i):
        for p in prange(npos_j):
            N[d,p]=np.argmin(d2[d*apos_i:(d+1)*apos_i,p*apos_j:(p+1)*apos_j])
            dmin2[d,p]=d2[d*apos_i+int(N[d,p]/apos_j),p*apos_j+np.mod(N[d,p],apos_j)]
            mol_disp[d,p,:]=disp[d*apos_i+int(N[d,p]/apos_j), \
                p*apos_j+np.mod(N[d,p],apos_j),:]
    dmin=np.sqrt(dmin2) 
    return dmin,mol_disp

@njit(parallel=True)
def get_pbcdisp_mols(pos, box, atoms1, atoms2, pbc): #@
    """Calculate the displacement required to move atoms2 to minimize distance
        between groups of atoms2 and atoms1

    While get_pbc_mindist() can be used to calculated the required displacement,
    get_pbcdisp_mols is faster because square roots are not performed.
    Parameters
    ----------
    pos: 2D numpy array of floats
       See pos_i or pos_j in distance()
    box: numpy array of floats
       See get_pbc_mindist()
    atoms1: list of int
       Global ID of atoms in a group (1). 
    atoms2: list of int
       Global ID of atoms in a group (2)
    pbc: list of integer
       See get_pbc_mindist()

    Returns
    -------
    min_disp: array of ints
        Contains the displacement of atoms2 needed to minimize its distance with
        atoms1. Similar to mol_disp in get_pbc_mindist() 
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
    d2=np.sum(np.square(dr2),axis=2)
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
       See dist_matrix()
    disp: numpy array of integers
       See min_disp in get_pbcdisp_mols()
    atoms: numpy array of integers
       See atoms in get_cen()
 
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
       See pos_i or pos_j in distance()
    box: numpy array of floats
       See box in get_pbc_mindist()
    atoms: List of integers
       See atoms in get_cen()
    pbc: list of integer
       See get_pbc_mindist()

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

