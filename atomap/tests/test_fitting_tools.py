import unittest
import numpy as np
import atomap.fitting_tools as ft


class test_ODR_fitter(unittest.TestCase):

    def test_simple(self):
        x = np.arange(0, 10, 1)
        y = np.arange(0, 20, 2)
        beta = ft.ODR_linear_fitter(x, y)
        self.assertAlmostEqual(beta[0], 2, places=5)
        self.assertAlmostEqual(beta[1], 0, places=5)

    def test_flat(self):
        x = np.arange(0, 10, 1)
        y = np.full_like(x, 1)
        beta = ft.ODR_linear_fitter(x, y)
        self.assertAlmostEqual(beta[0], 0, places=5)
        self.assertAlmostEqual(beta[1], 1, places=5)

    def test_vertical(self):
        y = np.arange(0, 10, 1)
        x = np.full_like(y, 5)
        beta = ft.ODR_linear_fitter(x, y)
        y_vector = [0, ft.linear_fit_func(beta, 5)]
        x_vector = [1, 0]
        dot = np.dot(x_vector, y_vector)
        self.assertEqual(dot, 0.0)


class test_find_distance_point_line(unittest.TestCase):

    def test_simple(self):
        x_list, y_list = [0], [0]
        line = [0, 1]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        self.assertTrue(d[0] == 1.)

    def test_negative(self):
        x_list, y_list = [0], [0]
        line = [0, -1]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        self.assertTrue(d == -1.)

    def test_vertical_line(self):
        x_list, y_list = [0, 2], [1, 1]
        line = [400000, -400000]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        self.assertAlmostEqual(d[0], -1., places=4)
        self.assertAlmostEqual(d[1], 1., places=4)

    def test_60_degreees(self):
        x_list, y_list = [0], [0]
        line = [-np.tan(np.radians(30)), 3]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        self.assertAlmostEqual(d[0], 2.598, places=2)

    def test_45_degreees(self):
        x_list, y_list = [0], [0]
        line = [1, -3]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        self.assertAlmostEqual(d[0], -3*np.sin(np.radians(45)), places=2)

    def test_on_line(self):
        x_list, y_list = [1], [1]
        line = [1, 0]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        self.assertAlmostEqual(d[0], 0, places=2)
