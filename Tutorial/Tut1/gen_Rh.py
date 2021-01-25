import sys
sys.path.insert(0,'../../')
import NP_analysis as NPa

NPa.import_constants('.')
#NPa.update_gros(251,inname='GROs/DP',outname='New_GROs/New',move_method='mindist')
NPa.calc_Rh_Rg(251,'Rh1.dat','Rg1.dat','mass.dat',infile='New_GROs/New')
