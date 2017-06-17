import unittest
import numpy as np
from atomap.testing_tools import make_artifical_atomic_signal
from atomap.sublattice import Sublattice
from atomap.initial_position_finding import _find_dumbbell_vector


class test_fit_atom_positions_with_gaussian_model(unittest.TestCase):

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
        vector = _find_dumbbell_vector(s, 4)
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
        vector = _find_dumbbell_vector(s, 4)
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
        vector = _find_dumbbell_vector(s, 4)
        self.assertAlmostEqual(abs(vector[0]), 3., places=7)
        self.assertAlmostEqual(abs(vector[1]), 3., places=7)
