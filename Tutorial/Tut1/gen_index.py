import sys
sys.path.insert(0,'../../')
import NP_analysis as NPa

NPa.import_constants('.')
print("The system contains % d DNAs and % d PEIs" %(NPa.ndna,NPa.npei))
NPa.write_index_mol('index_mol.ndx')
