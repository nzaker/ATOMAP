import unittest
import numpy as np
from atomap.testing_tools import MakeTestData
from atomap.initial_position_finding import (
        find_dumbbell_vector, _get_dumbbell_arrays,
        make_atom_lattice_dumbbell_structure)
from atomap.atom_finding_refining import get_atom_positions


class test_find_dumbbell_vector(unittest.TestCase):

    def test_5_separation_x(self):
        test_data = MakeTestData(200, 200)
        x0, y0 = np.mgrid[10:200:20, 10:200:20]
        x1, y1 = np.mgrid[16:200:20, 10:200:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        vector = find_dumbbell_vector(test_data.signal, 4)
        self.assertAlmostEqual(abs(vector[0]), 6., places=7)
        self.assertAlmostEqual(abs(vector[1]), 0., places=7)

    def test_5_separation_y(self):
        test_data = MakeTestData(200, 200)
        x0, y0 = np.mgrid[10:200:20, 10:200:20]
        x1, y1 = np.mgrid[10:200:20, 16:200:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        vector = find_dumbbell_vector(test_data.signal, 4)
        self.assertAlmostEqual(abs(vector[0]), 0., places=7)
        self.assertAlmostEqual(abs(vector[1]), 6., places=7)

    def test_3x_3y_separation(self):
        test_data = MakeTestData(200, 200)
        x0, y0 = np.mgrid[10:200:20, 10:200:20]
        x1, y1 = np.mgrid[13:200:20, 13:200:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        vector = find_dumbbell_vector(test_data.signal, 4)
        self.assertAlmostEqual(abs(vector[0]), 3., places=7)
        self.assertAlmostEqual(abs(vector[1]), 3., places=7)


class test_get_dumbbell_arrays(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(230, 230)
        x0, y0 = np.mgrid[20:210:20, 20:210:20]
        x1, y1 = np.mgrid[26:210:20, 20:210:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        self.s = test_data.signal

    def test_simple_running(self):
        s = self.s
        vector = find_dumbbell_vector(s, 4)
        position_list = get_atom_positions(s, 14)
        dumbbell_array0, dumbbell_array1 = _get_dumbbell_arrays(
                s, position_list, vector)
        self.assertEqual(len(dumbbell_array0), 100)
        self.assertEqual(len(dumbbell_array1), 100)


class test_make_atom_lattice_dumbbell_structure(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(230, 230)
        x0, y0 = np.mgrid[20:210:20, 20:210:20]
        x1, y1 = np.mgrid[26:210:20, 20:210:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        self.s = test_data.signal

    def test_simple_running(self):
        s = self.s
        vector = find_dumbbell_vector(s, 4)
        position_list = get_atom_positions(s, separation=13)
        atom_lattice = make_atom_lattice_dumbbell_structure(
                s, position_list, vector)
        self.assertEqual(len(atom_lattice.sublattice_list), 2)
        sublattice0 = atom_lattice.sublattice_list[0]
        sublattice1 = atom_lattice.sublattice_list[1]
        self.assertEqual(len(sublattice0.atom_list), 100)
        self.assertEqual(len(sublattice1.atom_list), 100)
