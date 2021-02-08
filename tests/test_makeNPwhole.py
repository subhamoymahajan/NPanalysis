import sys
sys.path.insert(0,'../')
import NPanalysis as NPa
import numpy as np
import unittest
import networkx as nx
import os
class TestConnections(unittest.TestCase):

    def setUp(self):
        self.connected=nx.read_gpickle('data/connected.pickle')       
        os.system('cp data/constants.pickle .')

    def test_connMat(self):
        #Calculate connection matrix from gromacs generated mindist.
        os.system('tar -xzf data/Whole.tar.gz')
        NPa.gmx.make_NPwhole(inGRO='Whole/DP',outGRO='New/New', \
            cluster_pickle='data/cluster.pickle', ndx_pickle='data/molndx.pickle', \
            connected_pickle='data/connected.pickle')

        NPa.connMat.gro2connected(inGRO='New/New',ndx_pickle='data/molndx.pickle', \
            connected_pickle='connected_2.pickle', time_shift=0, time_fac=1, \
            mindist_pickle='mindist_2.pickle', time_pickle='time_2.pickle')
        os.system('rm -r Whole')
        Nt,Nd,Np=(self.connected).shape
        connected_2=nx.read_gpickle('connected_2.pickle')
        Nt_2,Nd_2,Np_2=connected_2.shape
        self.assertEqual(Nt,Nt_2)
        self.assertEqual(Nd,Nd_2)
        self.assertEqual(Np,Np_2)
        
        for t in range(Nt):
            for d in range(Nd):
                for p in range(Np):
                    self.assertEqual(self.connected[t,d,p],connected_2[t,d,p])
        os.system('rm *.pickle')        
        os.system('rm -r New')

if __name__ == '__main__':
    unittest.main()
