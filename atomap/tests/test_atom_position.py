import unittest
from numpy import pi
import math
from atomap.atom_position import Atom_Position


class test_create_atom_position_object(unittest.TestCase):

    def test_position(self):
        atom_x, atom_y = 10, 20
        atom_position = Atom_Position(atom_x, atom_y)
        self.assertEqual(atom_position.pixel_x, 10)
        self.assertEqual(atom_position.pixel_y, 20)

    def test_sigma(self):
        sigma_x, sigma_y = 2, 3
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        self.assertEqual(atom_position.sigma_x, sigma_x)
        self.assertEqual(atom_position.sigma_y, sigma_y)

        sigma_x, sigma_y = -5, -6
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        self.assertEqual(atom_position.sigma_x, abs(sigma_x))
        self.assertEqual(atom_position.sigma_y, abs(sigma_y))

    def test_amplitude(self):
        amplitude = 30
        atom_position = Atom_Position(1, 2, amplitude=amplitude)
        self.assertEqual(atom_position.amplitude_gaussian, amplitude)

    def test_rotation(self):
        rotation0 = 0.0
        atom_position = Atom_Position(1, 2, rotation=rotation0)
        self.assertEqual(atom_position.rotation, rotation0)

        rotation1 = math.pi/2
        atom_position = Atom_Position(1, 2, rotation=rotation1)
        self.assertEqual(atom_position.rotation, rotation1)

        rotation2 = math.pi
        atom_position = Atom_Position(1, 2, rotation=rotation2)
        self.assertEqual(atom_position.rotation, 0)

        rotation3 = math.pi*3/2
        atom_position = Atom_Position(1, 2, rotation=rotation3)
        self.assertEqual(atom_position.rotation, math.pi/2)

        rotation4 = math.pi*2
        atom_position = Atom_Position(1, 2, rotation=rotation4)
        self.assertEqual(atom_position.rotation, 0)


class test_atom_position_object_tools(unittest.TestCase):

    def setUp(self):
        self.atom_position = Atom_Position(1, 1)

    def test_get_atom_angle(self):
        atom_position0 = Atom_Position(1, 2)
        atom_position1 = Atom_Position(3, 1)
        atom_position2 = Atom_Position(1, 0)
        atom_position3 = Atom_Position(5, 1)
        atom_position4 = Atom_Position(2, 2)

        angle90 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position1)
        angle180 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position2)
        angle0 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position3)
        angle45 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position4)

        self.assertAlmostEqual(angle90, pi/2, 7)
        self.assertAlmostEqual(angle180, pi, 7)
        self.assertAlmostEqual(angle0, 0, 7)
        self.assertAlmostEqual(angle45, pi/4, 7)

    def test_as_gaussian(self):
        x, y, sx, sy, A, r = 10., 5., 2., 3.5, 9.9, 1.5
        atom_position = Atom_Position(
                x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        g = atom_position.as_gaussian()
        self.assertEqual(g.centre_x.value, x)
        self.assertEqual(g.centre_y.value, y)
        self.assertEqual(g.sigma_x.value, sx)
        self.assertEqual(g.sigma_y.value, sy)
        self.assertEqual(g.A.value, A)
        self.assertEqual(g.rotation.value, r)
