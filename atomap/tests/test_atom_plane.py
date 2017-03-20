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
                atom_list, (1, 2), atom_lattice)
