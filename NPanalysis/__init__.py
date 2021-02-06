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

from . import connMat
from . import cluster
from . import radius
from . import geometry
from . import gmx

def pickle_constants(filename, sep=' '):
    """Imports constants required for nanoparticle analysis from a file and 
       pickles it 
    
    Parameters
    ----------
    path: str
         the relative path of constants file
    sep: str
         the separator of arryas such as 'phos_ids', 'nitr_ids' 
    Writes
    ------
    [filename].pickle: dictionary
        A dictionary of constants.
        ndna: int
            Total number of DNAs in the system
        npei: int
            Total number of PEIs in the system
        adna: int
            Total number of atoms in a DNA molecule
        apei: int
            Total number of atoms in a PEI molecule
        sdna: int
            Index of first DNA atom in .ndx file
        spei: int
            Index of first PEI atom in .ndx file
        Qdna: float
            Charge of a DNA molecule
        Qpei: float 
            Charge of a PEI molecule
        dna_name: str
            Name of the DNA in the .ndx file
        pei_name: str
            Name of the PEI in the .ndx file
        phos_ids: List of int
            Local atom ID of phosphoros beads/atoms. Since this is negatively
            charged, its a primary point of contact. Currently not used.
        nitr_ids: List of int 
            Local atom ID of amine beads/atoms. Since this is positively
            charged, its a primary point of contact. Currently not used.
        contact_dist: float
            Distance below which two molecules/atoms are considered bound.
    """
    print("Writing: constants.pickle\n")
    constants={}
    f=open(filename,'r')
    for lines in f:
        if len(lines)==0: #ignore empty lines
            continue
        if lines[0]=='#' or lines[0]=='@': # ignore comments
            continue
        foo=lines.split('=')
        foo[0]=foo[0].strip()
        if len(foo)<2:
            continue
        if foo[0] in ["ndna", "npei","adna","apei","sdna","spei"]: #integers
            constants[foo[0]]=int(foo[1])
        elif foo[0] in ["contact_dist","Qpei","Qdna"]: #floats
            constants[foo[0]]=float(foo[1])
        elif foo[0] in ["phos_ids","nitr_ids"]: 
        #array of integers separated by spaces
            foo2=(foo[1].strip()).split(sep)
            foo2=[int(x) for x in foo2]
            constants[foo[0]]=foo2
        elif foo[0] in ["dna_name","pei_name"]: #strings
            constants[foo[0]]=foo[1].strip()
        elif foo[0] == "pbc":
            pbc=[0,0,0]
            if 'x' in foo[1]:
                pbc[0]=1
            if 'y' in foo[1]:
                pbc[1]=1
            if 'z' in foo[1]:
                pbc[2]=1
            pbc=np.array(pbc) 
            constants['pbc']=pbc

    print("CONSTANTS")
    print("---------")
    for keys in constants:
        print(keys+" = "+str(constants[keys]))
    print("---------\n")

    nx.write_gpickle(constants,'constants.pickle')
    f.close()

