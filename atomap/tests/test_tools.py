import unittest
from atomap.tools import array2signal1d, array2signal2d, Fingerprinter
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import atomap.testing_tools as tt
import numpy as np


class TestArray2Signal(unittest.TestCase):

    def test_array2signal2d(self):
        array = np.arange(100).reshape(10, 10)
        scale = 0.2
        s = array2signal2d(array, scale=scale)
        self.assertEqual(s.axes_manager[0].scale, scale)
        self.assertEqual(s.axes_manager[1].scale, scale)
        self.assertEqual(s.axes_manager.shape, array.shape)

    def test_array2signal1d(self):
        array = np.arange(100)
        scale = 0.2
        s = array2signal1d(array, scale=scale)
        self.assertEqual(s.axes_manager[0].scale, scale)
        self.assertEqual(s.axes_manager.shape, array.shape)


class TestFingerprinter(unittest.TestCase):

    def test_running_simple(self):
        xN, yN, n = 3, 3, 60
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        clusters = (xN*2+1)*(yN*2+1)-1
        self.assertEqual(fp.cluster_centers_.shape[0], clusters)

    def test_running_advanced(self):
        xN, yN, xS, yS, n = 2, 2, 9, 12, 70
        tolerance = 0.5
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, xS=xS, yS=yS, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        bool_list = []
        for ix in list(range(-xN, xN + 1)):
            for iy in list(range(-yN, yN + 1)):
                if (ix == 0) and (iy == 0):
                    pass
                else:
                    for point in point_list:
                        x, y = ix*xS, iy*yS
                        xD, yD = abs(x-point[0]), abs(y-point[1])
                        if (xD < tolerance) and (yD < tolerance):
                            bool_list.append(True)
                            break
        clusters = (xN*2+1)*(yN*2+1)-1
        self.assertEqual(len(bool_list), clusters)


class test_remove_atoms_from_image_using_2d_gaussian(unittest.TestCase):

    def setUp(self):
        test_data = tt.MakeTestData(520, 520)
        x, y = np.mgrid[10:510:8j, 10:510:8j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        sublattice = test_data.sublattice
        sublattice.find_nearest_neighbors()
        self.sublattice = sublattice

    def test_running(self):
        sublattice = self.sublattice
        remove_atoms_from_image_using_2d_gaussian(
                sublattice.image,
                sublattice,
                percent_to_nn=0.40)
