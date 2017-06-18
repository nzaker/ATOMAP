import matplotlib
matplotlib.use('Agg')
import unittest
from atomap.tools import array2signal1d, array2signal2d, Fingerprinter
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import atomap.testing_tools as tt
import numpy as np
from atomap.atom_finding_refining import get_atom_positions
from atomap.sublattice import Sublattice


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
        x, y = np.mgrid[0:500:20j,0:500:20j]
        x, y = x.flatten(), y.flatten()
        s, g_list = tt.make_artifical_atomic_signal(x, y, image_pad=10)
        atom_positions = get_atom_positions(
                signal=s,
                separation=10,
                threshold_rel=0.02,
                )
        sublattice = Sublattice(
                atom_position_list=atom_positions,
                image=s.data)
        sublattice._find_nearest_neighbors()
        self.sublattice = sublattice

    def test_running(self):
        sublattice = self.sublattice
        subtracted_image = remove_atoms_from_image_using_2d_gaussian(
                sublattice.image,
                sublattice,
                percent_to_nn=0.40)
