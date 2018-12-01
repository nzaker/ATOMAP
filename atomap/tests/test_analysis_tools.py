import numpy as np
from pytest import approx
import atomap.api as am
import atomap.analysis_tools as an


class TestGetNeighborMiddlePosition:

    def test_simple(self):
        sublattice = am.dummy_data.get_simple_cubic_sublattice()
        sublattice.construct_zone_axes()
        za0 = sublattice.zones_axis_average_distances[0]
        za1 = sublattice.zones_axis_average_distances[1]
        atom = sublattice.atom_list[0]
        middle_pos = an.get_neighbor_middle_position(atom, za0, za1)
        atom1 = sublattice.atom_list[1]
        pos = np.mean((atom1.pixel_y, atom.pixel_y))
        assert approx(middle_pos) == (pos, pos)


class TestGetMiddlePositionList:

    def test_simple(self):
        sublattice = am.dummy_data.get_simple_cubic_sublattice()
        sublattice.construct_zone_axes()
        za0 = sublattice.zones_axis_average_distances[0]
        za1 = sublattice.zones_axis_average_distances[1]
        middle_pos_list = an.get_middle_position_list(sublattice, za0, za1)
        assert len(middle_pos_list) == (19 * 19)


class TestGetVectorShiftList:

    def test_simple(self):
        sublattice = am.Sublattice([[10., 10.], ], image=np.zeros((20, 20)))
        position_list = [[5., 5.], ]
        vector_list = an.get_vector_shift_list(sublattice, position_list)
        assert vector_list[0] == (5., 5., -5., -5.)

    def test_multiple(self):
        atom_list = np.arange(0, 100, 10).reshape(5, 2)
        sublattice = am.Sublattice(atom_list, image=np.zeros((20, 20)))
        position_list = np.arange(1, 101, 10).reshape(5, 2)
        vector_list = an.get_vector_shift_list(sublattice, position_list)
        for vector in vector_list:
            assert vector[2:] == (1., 1.)
