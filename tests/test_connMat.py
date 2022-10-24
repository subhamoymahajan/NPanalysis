import NPanalysis as NPa
import numpy as np
import unittest
import networkx as nx
import os

class TestConnections(unittest.TestCase):

    def setUp(self):
        self.mindist=nx.read_gpickle('data/mindist.pickle')     
        os.system('cp data/constants.pickle .')
    def test_connMat(self):
        #Calculate connection matrix from gromacs generated mindist.
        Nt,Nd,Np=(self.mindist).shape
        NPa.connMat.mindist2connected(connected_pickle='connected_1.pickle',\
            mindist_pickle='data/mindist.pickle')
   
        os.system('tar -xzf data/Whole.tar.gz')
        NPa.connMat.gro2connected(inGRO='Whole/DP',ndx_pickle='data/molndx.pickle', \
            connected_pickle='connected_2.pickle', time_shift=0, time_fac=1, \
            mindist_pickle='mindist_2.pickle', time_pickle='time_2.pickle')
        os.system('rm -r Whole')
        mindist_2=nx.read_gpickle('mindist_2.pickle')
        Nt_2,Nd_2,Np_2=mindist_2.shape
        self.assertEqual(Nt,Nt_2)
        self.assertEqual(Nd,Nd_2)
        self.assertEqual(Np,Np_2)
        
        for t in range(Nt):
            for d in range(Nd):
                for p in range(Np):
                    foo=abs(self.mindist[t,d,p]-mindist_2[t,d,p])
                    self.assertLessEqual(foo,np.sqrt(3)*1E-3, \
                        msg='t={0} d={1} p={2}'.format(t,d,p))

        
        connected_1=nx.read_gpickle('connected_1.pickle')
        Nt_3,Nd_3,Np_3=connected_1.shape
        self.assertEqual(Nt,Nt_3)
        self.assertEqual(Nd,Nd_3)
        self.assertEqual(Np,Np_3)
        

        connected_2=nx.read_gpickle('connected_2.pickle')
        Nt_4,Nd_4,Np_4=connected_2.shape
        self.assertEqual(Nt,Nt_4)
        self.assertEqual(Nd,Nd_4)
        self.assertEqual(Np,Np_4)

        for t in range(Nt):
            for d in range(Nd):
                for p in range(Np):
                    self.assertEqual(connected_1[t,d,p],connected_2[t,d,p])
        os.system('rm *.pickle')        

if __name__ == '__main__':
    unittest.main()
