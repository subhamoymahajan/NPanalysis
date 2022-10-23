import NPanalysis as NPa
import networkx as nx
import numpy as np
print("Part 1: Calculating Role Conversion\n")
NPa.connMat.get_role_conversion(connected_pickle='connected.pickle', main_mol=1, \
    outname='role_conv.dat',sep=',')

print("Part 2: NP Moment of Inertia (amu.nm^2)\n")
pos,box,text=NPa.gmx.read_gro('NPwhole/DP250.gro')
clusters=nx.read_gpickle('cluster.pickle')
constants=nx.read_gpickle('constants.pickle')
dname=constants['dna_name']
pname=constants['pei_name']
ndx=nx.read_gpickle('molndx.pickle')
atoms=NPa.gmx.get_NPatomIDs(clusters[-1],ndx,dname,pname, main_mol=0, NP_ID=1)
mass=nx.read_gpickle('mass.pickle')
MOI=NPa.radius.calc_MOI(pos, atoms, mass)
print(MOI)

print("Part 3: NP shape descriptors of last timestep.")
#cluster[-1] is clusters in the last timestep. cluster[-1][1] is the second NP in last timestep.
NPa.radius.NP_shape(clusters[-1][1],inGRO='NPwhole/DP250.gro', \
                    ndx_pickle='molndx.pickle')
print("Part 4: Mindist between Charged particles")
NPa.gmx.gen_index_charge('Qndx.pickle',prefix='Q')
NPa.gmx.write_index_mol('Qndx.ndx',ndx_pickle='Qndx.pickle',prefix='Q')
NPa.connMat.gro2connected(inGRO='NPwhole/DP',connected_pickle='Qconn.pickle',
    ndx_pickle='Qndx.pickle', time_shift=0, time_fac=4E-6, prefix='Q', 
    time_pickle='time.pickle', mindist_pickle='Qmind.pickle')
Qmin_d=nx.read_gpickle('Qmind.pickle')
print('Mindist for DNA5-PEI20 (between charged particles)')
print(Qmin_d[:,5,58])

