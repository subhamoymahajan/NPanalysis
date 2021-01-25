import sys
sys.path.insert(0,'../../')
import NP_analysis as NPa

#Import constants
NPa.import_constants('.') # constants.py is in the current directory.
#Testing important constants
print('The system has %d DNAs and %d PEIs' %(NPa.ndna,NPa.npei))

#Calculating Connection matrix
#Check number of time steps in mindist file mindist/mindist0-0.xvg using the command:
#grep -v '@' mindist/mindist0-0.xvg | grep -v '#' | wc -l
#For this tutorial the number of time steps is 251 
NPa.conv_mindist2connected(251,mindist_loc='mindist/')
NPa.get_PEI_roles(50,251)
NPa.get_PEI_roles2(50,251)

#Calculate clusters
NPa.gen_clusters(251)
NPa.w2f_cluster(outheader='cluster/cluster')

#Average size
NPa.gen_avgsize(50,'avgsize.dat',251,0.002) # The trajectory is 500ns, so each dt is 500/250 = 2ns. Therefore, 0.002 is dt in microseconds

#Average number of NP and charge of NP as a function of size
NPa.gen_ncNP_s(251,5,'num_charge_NPs_size.dat')

#Create data files for calculation of radius of gyration and hydrodynamic radius


    
