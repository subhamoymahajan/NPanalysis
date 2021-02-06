import sys
sys.path.insert(0,'../../')
import NPanalysis as NPa
##0.Preping system
# write the file 'constants.dat'
# get .gro files using "gmx trjconv -pbc whole -sep -f md_1.xtc -s md_1.tpr -o \
#     GROs/DP.gro"
# get mass infromation "gmx dump -s md_1.tpr > trj.dump"
##1.Reading constants
print("Step 1: Storing Constants\n")
NPa.pickle_constants('constants.dat', sep=' ') # constants is in the current directory.
NPa.gmx.gen_index_mol('molndx.pickle')
NPa.gmx.pickle_mass(filename='tpr.dump',mass_pickle='mass.pickle')

#2.Getting connection matrix from .gro files and finding change required to 
#  make nanoparticles whole.
print("Step 2: Calculating Connection Matrix\n")
NPa.connMat.gro2connected(inGRO='Whole/DP', time_pickle='time.pickle',  
    connected_pickle='connected.pickle', time_fac=4*1E-6, time_shift=0, 
    mindist_pickle='mindist.pickle', ndx_pickle='molndx.pickle')
NPa.connMat.get_roles(50, connected_pickle='connected.pickle', mol=1, 
    time_pickle='time.pickle', outname='PEI_roles.dat',sep=',')
NPa.connMat.get_roles2(50, connected_pickle='connected.pickle', mol=1, 
    time_pickle='time.pickle', outname='PEI_roles2.dat',sep=',')

##3.Calculate clusters
print("Step 3: Calculating Clusters\n")
NPa.cluster.gen_clusters(connected_pickle='connected.pickle', 
    cluster_pickle='cluster.pickle')
#Average size 
NPa.cluster.gen_avgsize(50,'avgsize.dat', cluster_pickle='cluster.pickle', 
     time_pickle='time.pickle', main_mol=0, sep=',')
#Average number of NP and charge of NP as a function of size
NPa.cluster.gen_ncNP_s(3,'num_charge_NPs_size.dat', main_mol=0, sep=' ', 
    cluster_pickle='cluster.pickle')

##4.Calculate Radius
##Make nanoparticles whole
#print("Step 4: Making Nanoparticles Whole and calculating Radii\n")
NPa.gmx.make_NPwhole(inGRO='Whole/DP',outGRO='NPwhole/DP', 
    cluster_pickle='cluster.pickle', connected_pickle='connected.pickle', 
    ndx_pickle='molndx.pickle')

##Calculate radius at each timestep.
NPa.radius.calc_Rh_Rg(Rh_pickle='Rh.pickle', Rg2_pickle='Rg2.pickle', sep=',', 
    inGRO='NPwhole/DP', main_mol=0, mass_pickle='mass.pickle', 
    cluster_pickle='cluster.pickle', ndx_pickle='molndx.pickle')

#Get average radius
NPa.radius.gen_rad_avg('Rh.pickle', 50, 'avg_Rh.dat', sep=',', sqrt=False, 
    time_pickle='time.pickle')
NPa.radius.gen_rad_avg('Rg2.pickle', 50, 'avg_Rg.dat', sep=',',sqrt=True, 
    time_pickle='time.pickle')
#Get average radius per unit size.
NPa.radius.gen_rad_per_size('Rh.pickle', 'Rh_size.dat', 200, 251, sep=',', 
     main_mol=0, cluster_pickle='cluster.pickle', sqrt=False)
NPa.radius.gen_rad_per_size('Rg2.pickle', 'Rg_size.dat', 200, 251, sep=',', 
     main_mol=0, cluster_pickle='cluster.pickle', sqrt=True)
###5.Clean up.
## Convert .gro files with whole nanoparticles into xtc file. 
## Delete .gro files
