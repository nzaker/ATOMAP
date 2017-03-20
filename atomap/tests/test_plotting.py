import matplotlib
matplotlib.use('Agg')
import unittest
from atomap.plotting import _make_atom_position_marker_list
from atomap.atom_position import Atom_Position
from atomap.atom_plane import Atom_Plane
from atomap.atom_lattice import Atom_Lattice
from atomap.plotting import (
        _make_single_atom_plane_marker_list,
        _make_atom_planes_marker_list)


class test_MakeAtomPositionMarkerList(unittest.TestCase):

    def test_make_atom_position_marker_list(self):
        atom_position_list = []
        for i in range(20):
            atom_position = Atom_Position(i, i)
            atom_position_list.append(atom_position)
        marker_list = _make_atom_position_marker_list(
                atom_position_list,
                scale=0.2,
                markersize=30,
                color='black',
                add_numbers=True)
        self.assertEqual(len(marker_list), 40)


class test_atom_plane_marker_plotting(unittest.TestCase):

    def setUp(self):
        atom_lattice = Atom_Lattice()
        atom_plane_list = []
        for i in range(2):
            atom_list = [
                    Atom_Position(1, 2),
                    Atom_Position(2, 4),
                    ]
            zone_vector = (1, 2)
            atom_list[0]._start_atom = [zone_vector]
            atom_list[1]._end_atom = [zone_vector]
            atom_plane = Atom_Plane(
                    atom_list, (1, 2), atom_lattice)
            atom_plane_list.append(atom_plane)
        self.atom_plane_list = atom_plane_list

    def test_single_atom_plane_marker(self):
        atom_plane = self.atom_plane_list[0]
        marker_list = _make_single_atom_plane_marker_list(atom_plane)
        self.assertEqual(len(marker_list), 1)

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
