import unittest
import numpy as np
from atomap.atom_finding_refining import _make_mask_from_positions


class test_make_mask_from_positions(unittest.TestCase):

    def test_radius_1(self):
        x, y, r = 10, 20, 1
        pos = [[x, y]]
        rad = [r]
        mask = _make_mask_from_positions(pos, rad, (40, 40))
        self.assertEqual(mask.sum(), 5.)
        mask[x, y] = False
        mask[x+r, y] = False
        mask[x-r, y] = False
        mask[x, y+1] = False
        mask[x, y-1] = False
        self.assertFalse(mask.any())

    def test_2_positions_radius_1(self):
        x0, y0, x1, y1, r = 10, 20, 20, 30, 1
        pos = [[x0, y0], [x1, y1]]
        rad = [r, r]
        mask = _make_mask_from_positions(pos, rad, (40, 40))
        self.assertEqual(mask.sum(), 10.)
        mask[x0, y0] = False
        mask[x0+r, y0] = False
        mask[x0-r, y0] = False
        mask[x0, y0+1] = False
        mask[x0, y0-1] = False
        mask[x1, y1] = False
        mask[x1+r, y1] = False
        mask[x1-r, y1] = False
        mask[x1, y1+1] = False
        mask[x1, y1-1] = False
        self.assertFalse(mask.any())

    def test_radius_2(self):
        x, y, r = 10, 5, 2
        pos = [[x, y]]
        rad = [r]
        mask = _make_mask_from_positions(pos, rad, (40, 40))
        self.assertEqual(mask.sum(), 13.)

    def test_2_positions_radius_2(self):
        x0, y0, x1, y1, r = 5, 7, 17, 25, 2
        pos = [[x0, y0], [x1, y1]]
        rad = [r, r]
        mask = _make_mask_from_positions(pos, rad, (40, 40))
        self.assertEqual(mask.sum(), 26.)

    def test_wrong_input(self):
        x, y, r = 10, 5, 2
        pos = [[x, y]]
        rad = [r, r]
        mask = (pos, rad, (40, 40))
        self.assertRaises(
                ValueError,
                _make_mask_from_positions,
                position_list=pos,
                radius_list=rad,
                data_shape=(40, 40))
