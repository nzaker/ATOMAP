import matplotlib
matplotlib.use('Agg')
import unittest
from atomap.atom_plane import Atom_Plane
from atomap.atom_position import Atom_Position
from atomap.atom_lattice import Atom_Lattice
from atomap.plotting import (
        _make_single_atom_plane_marker_list,
        _make_atom_planes_marker_list)


class test_create_atom_plane_object(unittest.TestCase):

    def test_create_atom_plane_object(self):
        atom_list = [
                Atom_Position(1, 2),
                Atom_Position(2, 4),
                ]
        zone_vector = (1, 2)
        atom_list[0]._start_atom = [zone_vector]
        atom_list[1]._end_atom = [zone_vector]
        atom_lattice = Atom_Lattice()
        atom_plane = Atom_Plane(
                atom_list, zone_vector, atom_lattice)


class test_atom_plane_properties(unittest.TestCase):

    def setUp(self):
        x, y = [1, 2], [2, 4]
        sX, sY, r = [3.1, 1.2], [2.2, 1.1], [0.5, 0.4]
        A_g = [10.2, 5.2]

        atom_list = [
                Atom_Position(x[0], y[0], sX[0], sY[0], r[0]),
                Atom_Position(x[1], y[1], sX[1], sY[1], r[1]),
                ]
        atom_list[0].amplitude_gaussian = A_g[0]
        atom_list[1].amplitude_gaussian = A_g[1]
        zone_vector = (1, 2)
        atom_list[0]._start_atom = [zone_vector]
        atom_list[1]._end_atom = [zone_vector]
        self.atom_plane = Atom_Plane(
                atom_list,
                zone_vector,
                Atom_Lattice())
        self.x, self.y, self.sX, self.sY, self.r = x, y, sX, sY, r
        self.A_g = A_g

    def test_x_position(self):
        self.assertEqual(self.atom_plane.x_position, self.x)

    def test_y_position(self):
        self.assertEqual(self.atom_plane.y_position, self.y)

    def test_sx_position(self):
        self.assertEqual(self.atom_plane.sigma_x, self.sX)

    def test_sy_position(self):
        self.assertEqual(self.atom_plane.sigma_y, self.sY)

    def test_sigma_average(self):
        sigma_ave = [0.5*(self.sX[0]+self.sY[0]), 0.5*(self.sX[1]+self.sY[1])]
        self.assertEqual(self.atom_plane.sigma_average, sigma_ave)

    def test_r_position(self):
        self.assertEqual(self.atom_plane.rotation, self.r)

    def test_ellipticity_position(self):
        elli = [self.sX[0]/self.sY[0], self.sX[1]/self.sY[1]]
        self.assertEqual(self.atom_plane.ellipticity, elli)

    def test_amplitude_gaussian(self):
        self.assertEqual(self.atom_plane.amplitude_gaussian, self.A_g)
