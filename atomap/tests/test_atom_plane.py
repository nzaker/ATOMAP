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
        atom_list[0].start_atom = [zone_vector]
        atom_list[1].end_atom = [zone_vector]
        atom_lattice = Atom_Lattice()
        atom_plane = Atom_Plane(
                atom_list, (1, 2), atom_lattice)


class test_atom_plane_object(unittest.TestCase):

    def setUp(self):
        atom_list = [
                Atom_Position(1, 2),
                Atom_Position(2, 4),
                ]
        zone_vector = (1, 2)
        atom_list[0].start_atom = [zone_vector]
        atom_list[1].end_atom = [zone_vector]
        atom_lattice = Atom_Lattice()
        atom_plane = Atom_Plane(
                atom_list, (1, 2), atom_lattice)
        self.atom_plane = atom_plane

    def test_single_atom_plane_marker(self):
        marker_list = _make_single_atom_plane_marker_list(self.atom_plane)
        self.assertEqual(len(marker_list), 1)

class test_atom_plane_list(unittest.TestCase):

    def setUp(self):
        atom_lattice = Atom_Lattice()
        atom_plane_list = []
        for i in range(2):
            atom_list = [
                    Atom_Position(1, 2),
                    Atom_Position(2, 4),
                    ]
            zone_vector = (1, 2)
            atom_list[0].start_atom = [zone_vector]
            atom_list[1].end_atom = [zone_vector]
            atom_plane = Atom_Plane(
                    atom_list, (1, 2), atom_lattice)
            atom_plane_list.append(atom_plane)
        self.atom_plane_list = atom_plane_list

    def test_make_atom_planes_marker_list_no_number(self):
        atom_plane_list = self.atom_plane_list
        marker_list = _make_atom_planes_marker_list(
                atom_plane_list, add_numbers=False)
        self.assertEqual(len(marker_list), 2)

    def test_make_atom_planes_marker_list(self):
        atom_plane_list = self.atom_plane_list
        marker_list = _make_atom_planes_marker_list(
                atom_plane_list, add_numbers=True)
        self.assertEqual(len(marker_list), 4)
