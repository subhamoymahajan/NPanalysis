import NPanalysis.geometry as geo
import numpy as np
import unittest

class TestGeometry(unittest.TestCase):

    def setUp(self):
        self.pos=np.array([[1,1,1],[2,2,2],[-1,2,-1],[-2,-2,3],[0,0,0]])
        self.pos1=np.array([[0.2,1.,0.],[0.2,1.2,0.],[0.2,1.2,0.4],
                            [1.9,1.,0.4],[1.7,1.1,0.5],[1.7,1.2,0.6],
                            [1.6,1.6,1.6],[1.5,1.4,1.3],[1.4,1.5,1.2]])
        self.pos2=np.array([[0.2,0.4,0],[0.2,0.4,0.2],[0.2,0.4,0.3],
                            [1.9,1.,0.],[1.7,1.2,0.],[1.7,1.2,0.2],
                            [1.2,1.2,1.2],[1.5,1.2,1.2],[1.3,1.4,1.4]])

    def test_distance2(self):
        self.assertAlmostEqual(geo.distance2(self.pos[0],self.pos[-1]),3)
        self.assertAlmostEqual(geo.distance2(self.pos[2],self.pos[3]),33)

    def test_distance(self):
        self.assertAlmostEqual(geo.distance(self.pos[1],self.pos[2]),np.sqrt(18))
        self.assertAlmostEqual(geo.distance(self.pos[3],self.pos[0]),np.sqrt(22))

    def test_distmat(self):
        mat=geo.dist_matrix(self.pos,maxd=None)
        self.assertAlmostEqual(mat[0,1],np.sqrt(3))
        self.assertAlmostEqual(mat[0,2],3)
        self.assertAlmostEqual(mat[0,3],np.sqrt(22))
        self.assertAlmostEqual(mat[0,4],np.sqrt(3))
        self.assertAlmostEqual(mat[1,2],np.sqrt(18))
        self.assertAlmostEqual(mat[1,3],np.sqrt(33))
        self.assertAlmostEqual(mat[1,4],np.sqrt(12))
        self.assertAlmostEqual(mat[2,3],np.sqrt(33))
        self.assertAlmostEqual(mat[2,4],np.sqrt(6))
        self.assertAlmostEqual(mat[3,4],np.sqrt(17))

    def test_distmat2(self):
        mat=geo.dist_matrix(self.pos,maxd=4.)
        self.assertAlmostEqual(mat[0,1],np.sqrt(3))
        self.assertAlmostEqual(mat[0,2],3.)
        self.assertAlmostEqual(mat[0,3],4.)
        self.assertAlmostEqual(mat[0,4],np.sqrt(3))
        self.assertAlmostEqual(mat[1,2],4.)
        self.assertAlmostEqual(mat[1,3],4.)
        self.assertAlmostEqual(mat[1,4],np.sqrt(12))
        self.assertAlmostEqual(mat[2,3],4.)
        self.assertAlmostEqual(mat[2,4],np.sqrt(6))
        self.assertAlmostEqual(mat[3,4],4.)

    def test_cen(self):
        cen=geo.get_cen(self.pos,np.array([0,1,2]))
        self.assertAlmostEqual(cen[0],2./3.)
        self.assertAlmostEqual(cen[1],5./3.)
        self.assertAlmostEqual(cen[2],2./3.)

    def test_pbcmindist(self):
        dmin,mol_disp=geo.get_pbc_mindist(self.pos1,self.pos2,np.array([2.,2.,2.]),3,3,3,3,[1,1,1])
        dmin_act=[[0.6,0.3,np.sqrt(1.13)],
                  [np.sqrt(0.46),np.sqrt(0.1),np.sqrt(0.4)],
                  [np.sqrt(1.16),np.sqrt(0.33),np.sqrt(0.05)]
                 ]
        disp_act=[ [[0.,0.,0.],[2.,0.,0.],[2.,0.,0.]],
                   [[-2.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
                   [[-2.,-2.,-2.],[0.,0.,-2.],[0.,0.,0.]]]
        for j in range(3):
            for i in range(3):
                self.assertAlmostEqual(dmin[i,j],dmin_act[i][j]) 
                for k in range(3):
                    if k==0 and i==2: #Two possible mindist with different mol_disp
                        continue
                    self.assertAlmostEqual(mol_disp[k,i,j],disp_act[k][i][j]) 

    def test_pbcdisp(self):
        disp_act=[ [[0.,0.,0.],[2.,0.,0.],[2.,0.,0.]],
                   [[-2.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
                   [[-2.,-2.,-2.],[0.,0.,-2.],[0.,0.,0.]]]
        pos=np.concatenate((self.pos1,self.pos2))
        for i in range(3):
            for j in range(3):
                if i==0 and j==2: #Two possible mindist with different mol_disp
                    continue
                disp=geo.get_pbcdisp_mols(pos,np.array([2.,2.,2.]),
                    np.arange(i*3,i*3+3),np.arange(j*3+9,j*3+12),[1,1,1])
                for k in range(3):
                    self.assertAlmostEqual(-disp[k],disp_act[i][j][k])

    # Tests are not performed for change_NPpos(), change_pos().
if __name__ == '__main__':
    unittest.main()
