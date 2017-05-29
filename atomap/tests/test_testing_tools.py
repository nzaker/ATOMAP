import os
import unittest
import matplotlib
matplotlib.use('Agg')
import atomap.testing_tools as tt
from numpy import mgrid


class TestMakeArtificalAtomicSignal(unittest.TestCase):
    def setUp(self):
        x, y = mgrid[0:50:5j,0:50:5j]
        self.x, self.y = x.flatten(), y.flatten()

    def test_running(self):
        x, y = self.x, self.y
        signal, gaussian_list = tt.make_artifical_atomic_signal(
                x=x, y=y)
        self.assertEqual(len(x), len(gaussian_list))


class TestMakeVectorTestGaussian(unittest.TestCase):
    def test_running(self):
        x, y, std, n = 10, 5, 0.5, 5000
        point_list = tt.make_vector_test_gaussian(
                x, y, standard_deviation=std, n=n)
        point_list_meanX = point_list[:,0].mean()
        point_list_meanY = point_list[:,1].mean()
        point_list_stdX = point_list[:,0].std()
        point_list_stdY = point_list[:,1].std()

        self.assertAlmostEqual(point_list_meanX, x, places=1)
        self.assertAlmostEqual(point_list_meanY, y, places=1)
        self.assertAlmostEqual(point_list_stdX, std, places=1)
        self.assertAlmostEqual(point_list_stdY, std, places=1)
        self.assertEqual(n, point_list.shape[0])


class TestMakeNnTestDataset(unittest.TestCase):
    def test_running(self):
        xN, yN, n = 4, 4, 60
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, n=n)

        total_point = n*(((2*xN)+1)*((2*yN)+1)-1)
        self.assertEqual(point_list.shape[0], total_point)
