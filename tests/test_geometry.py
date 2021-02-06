import sys
sys.path.insert(0,'../')
import NPanalysis.geometry as geo
import numpy as np
import unittest

class TestGeometry(unittest.TestCase):

    def setUp(self):
        self.pos=np.array([[1,1,1],[2,2,2],[-1,2,-1],[-2,-2,3],[0,0,0]])

    def test_distance2(self):
        self.assertAlmostEqual(geo.distance2(self.pos[0],self.pos[-1]),3)
        self.assertAlmostEqual(geo.distance2(self.pos[2],self.pos[3]),33)

    def test_distance(self):
        self.assertAlmostEqual(geo.distance(self.pos[1],self.pos[2]),np.sqrt(18))
        self.assertAlmostEqual(geo.distance(self.pos[3],self.pos[0]),np.sqrt(22))


if __name__ == '__main__':
    unittest.main()
