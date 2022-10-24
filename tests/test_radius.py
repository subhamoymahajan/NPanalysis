import NPanalysis as NPa
import numpy as np
import unittest

class TestRadius(unittest.TestCase):

    def setUp(self):
        pass

    def test_Rg(self):
        pos,box,tests=NPa.gmx.read_gro('data/DP0.gro')
        mass1=np.ones(10)*45.
        mass1[5]=72.
        mass2=np.ones(10)*45.
        mass2[2]=72.
        mass2[8]=72.

        Rg_1=NPa.radius.calc_Rg2(pos[:10],np.arange(10),mass1)
        self.assertAlmostEqual(round(np.sqrt(Rg_1),6),0.367412)
        Rg_2=NPa.radius.calc_Rg2(pos[10:20],np.arange(10),mass2)
        self.assertAlmostEqual(round(np.sqrt(Rg_2),6),0.421847)

    # Tests are not performed for change_NPpos(), change_pos().
if __name__ == '__main__':
    unittest.main()
