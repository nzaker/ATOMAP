import unittest
import numpy as np
from atomap.testing_tools import make_artifical_atomic_signal
from atomap.sublattice import Sublattice
from atomap.initial_position_finding import (
        find_dumbbell_vector, _get_dumbbell_arrays,
        make_atom_lattice_dumbbell_structure)
from atomap.atom_finding_refining import get_atom_positions


class test_find_dumbbell_vector(unittest.TestCase):

    def test_5_separation_x(self):
        x_list, y_list = [], []
        for x in range(10, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        for x in range(16, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        sigma_value = 1
        sigma = [sigma_value]*len(x_list)
        A = [50]*len(x_list)
        s, g_list = make_artifical_atomic_signal(
                x_list, y_list, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=0)
        vector = find_dumbbell_vector(s, 4)
        self.assertAlmostEqual(abs(vector[0]), 6., places=7)
        self.assertAlmostEqual(abs(vector[1]), 0., places=7)

    def test_5_separation_y(self):
        x_list, y_list = [], []
        for x in range(10, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        for x in range(10, 200, 20):
            for y in range(16, 200, 20):
                x_list.append(x)
                y_list.append(y)
        sigma_value = 1
        sigma = [sigma_value]*len(x_list)
        A = [50]*len(x_list)
        s, g_list = make_artifical_atomic_signal(
                x_list, y_list, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=0)
        vector = find_dumbbell_vector(s, 4)
        self.assertAlmostEqual(abs(vector[0]), 0., places=7)
        self.assertAlmostEqual(abs(vector[1]), 6., places=7)

    def test_3x_3y_separation(self):
        x_list, y_list = [], []
        for x in range(10, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        for x in range(13, 200, 20):
            for y in range(13, 200, 20):
                x_list.append(x)
                y_list.append(y)
        sigma_value = 1
        sigma = [sigma_value]*len(x_list)
        A = [50]*len(x_list)
        s, g_list = make_artifical_atomic_signal(
                x_list, y_list, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=0)
        vector = find_dumbbell_vector(s, 4)
        self.assertAlmostEqual(abs(vector[0]), 3., places=7)
        self.assertAlmostEqual(abs(vector[1]), 3., places=7)


class test_get_dumbbell_arrays(unittest.TestCase):

    def setUp(self):
        x_list, y_list = [], []
        for x in range(10, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        for x in range(16, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        sigma_value = 1
        sigma = [sigma_value]*len(x_list)
        A = [50]*len(x_list)
        s, g_list = make_artifical_atomic_signal(
                x_list, y_list, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=0)
        self.s = s

    def test_simple_running(self):
        s = self.s
        vector = find_dumbbell_vector(s, 4)
        position_list = get_atom_positions(s, 14)
        dumbbell_array0, dumbbell_array1 = _get_dumbbell_arrays(
                s, position_list, vector)
        self.assertEqual(len(dumbbell_array0), 64)
        self.assertEqual(len(dumbbell_array1), 64)


class test_make_atom_lattice_dumbbell_structure(unittest.TestCase):

    def setUp(self):
        x_list, y_list = [], []
        for x in range(10, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        for x in range(16, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        sigma_value = 1
        sigma = [sigma_value]*len(x_list)
        A = [50]*len(x_list)
        s, g_list = make_artifical_atomic_signal(
                x_list, y_list, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=0)
        self.s = s

    def test_simple_running(self):
        s = self.s
        vector = find_dumbbell_vector(s, 4)
        position_list = get_atom_positions(s, separation=13)
        atom_lattice = make_atom_lattice_dumbbell_structure(
                s, position_list, vector)
        self.assertEqual(len(atom_lattice.sublattice_list), 2)
        sublattice0 = atom_lattice.sublattice_list[0]
        sublattice1 = atom_lattice.sublattice_list[0]
        self.assertEqual(len(sublattice0.atom_list),64)
        self.assertEqual(len(sublattice1.atom_list),64)
