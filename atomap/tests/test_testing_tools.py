import os
import unittest
import matplotlib
matplotlib.use('Agg')
import atomap.testing_tools as tt
from numpy import mgrid


class TestMakeTestData(unittest.TestCase):

    def test_simple_init(self):
        tt.MakeTestData(100, 100)

    def test_get_signal(self):
        imX0, imY0 = 100, 100
        test_data0 = tt.MakeTestData(imX0, imY0)
        self.assertEqual(imX0, test_data0.image_x)
        self.assertEqual(imY0, test_data0.image_y)
        self.assertEqual(test_data0.signal.axes_manager.shape, (imX0, imY0))

        imX1, imY1 = 100, 39
        test_data1 = tt.MakeTestData(imX1, imY1)
        self.assertEqual(imX1, test_data1.image_x)
        self.assertEqual(imY1, test_data1.image_y)
        self.assertEqual(test_data1.signal.axes_manager.shape, (imX1, imY1))

        imX2, imY2 = 34, 65
        test_data2 = tt.MakeTestData(imX2, imY2)
        self.assertEqual(imX2, test_data2.image_x)
        self.assertEqual(imY2, test_data2.image_y)
        self.assertEqual(test_data2.signal.axes_manager.shape, (imX2, imY2))

    def test_add_atom(self):
        x, y, sx, sy, A, r = 10, 5, 5, 9, 10, 2
        td = tt.MakeTestData(50, 50)
        td.add_atom(x, y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        self.assertEqual(len(td.sublattice.atom_list), 1)
        atom = td.sublattice.atom_list[0]
        self.assertEqual(atom.pixel_x, x)
        self.assertEqual(atom.pixel_y, y)
        self.assertEqual(atom.sigma_x, sx)
        self.assertEqual(atom.sigma_y, sy)
        self.assertEqual(atom.amplitude_gaussian, A)
        self.assertEqual(atom.rotation, r)

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
