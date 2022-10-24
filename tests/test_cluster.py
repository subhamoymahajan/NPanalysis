import NPanalysis as NPa
import numpy as np
import unittest
import networkx as nx
import os
class TestConnections(unittest.TestCase):

    def setUp(self):
        self.connected=nx.read_gpickle('data/connected.pickle')      
        os.system('cp data/constants.pickle .') 
        constants=nx.read_gpickle('constants.pickle')
        self.ndna=constants['ndna']
        self.npei=constants['npei']
 
    def test_connMat(self):
        #Calculate connection matrix from gromacs generated mindist.
        Nt,Nd,Np=(self.connected).shape
        NPa.cluster.gen_clusters(connected_pickle='data/connected.pickle',\
            cluster_pickle='cluster_2.pickle')
        clusters=nx.read_gpickle('cluster_2.pickle')
        Nt_2=len(clusters)
        self.assertEqual(Nt,Nt_2) 
     
        for t in range(Nt):
            #Check free DNA
            self.assertEqual(clusters[t][-2][1][0],-1,msg='t=0'.format(t))
            self.assertEqual(len(clusters[t][-2][1]),1,msg='t=0'.format(t))
            #Check free PEI
            self.assertEqual(clusters[t][-1][0][0],-1,msg='t=0'.format(t))
            self.assertEqual(len(clusters[t][-1][0]),1,msg='t={0}'.format(t))
            
            nclust=len(clusters[t])-2
            for cid in range(nclust):
                for di in clusters[t][cid][0]:
                    #Check if all connected PEIs to a DNA are in the cluster
                    peis=np.where(self.connected[t,di,:]==True)[0]
                    for pi in peis:
                        self.assertTrue(pi in clusters[t][cid][1],msg='t={0}'.format(t))
                for pi in clusters[t][cid][1]:
                    #Check if all connected DNAs to a PEI are in the cluster
                    dnas=np.where(self.connected[t,:,pi]==True)[0]
                    for di in dnas:
                        self.assertTrue(di in clusters[t][cid][0],msg='t={0}'.format(t))

            ### Checking total number of DNAs and PEIs
            ndna_2=0
            npei_2=0
            for cid in range(nclust):
                ndna_2+=len(clusters[t][cid][0])      
                npei_2+=len(clusters[t][cid][1])
            ndna_2+=len(clusters[t][-2][0])      
            npei_2+=len(clusters[t][-1][1])      
            
            self.assertEqual(self.ndna,ndna_2,msg='t={0}'.format(t))
            self.assertEqual(self.npei,npei_2,msg='t={0}'.format(t))
        os.system('rm *.pickle')        

if __name__ == '__main__':
    unittest.main()
