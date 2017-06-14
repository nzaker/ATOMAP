import unittest
import numpy as np
from atomap.atom_finding_refining import (
        _make_mask_from_positions,_crop_mask_slice_indices,
        _find_background_value, _find_median_upper_percentile)

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


class test_crop_mask(unittest.TestCase):

    def test_radius_1(self):
        x, y, r = 10, 20, 1
        pos = [[x, y]]
        rad = [r]
        mask = _make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = _crop_mask_slice_indices(mask)
        self.assertEqual(x0, x-r)
        self.assertEqual(x1, x+r+1)
        self.assertEqual(y0, y-r)
        self.assertEqual(y1, y+r+1)
        mask_crop = mask[x0:x1, y0:y1]
        self.assertEqual(mask_crop.shape, (2*r+1, 2*r+1))

    def test_radius_2(self):
        x, y, r = 15, 10, 2
        pos = [[x, y]]
        rad = [r]
        mask = _make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = _crop_mask_slice_indices(mask)
        mask_crop = mask[x0:x1, y0:y1]
        self.assertEqual(mask_crop.shape, (2*r+1, 2*r+1))

    def test_radius_5(self):
        x, y, r = 15, 10, 5
        pos = [[x, y]]
        rad = [r]
        mask = _make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = _crop_mask_slice_indices(mask)
        mask_crop = mask[x0:x1, y0:y1]
        self.assertEqual(mask_crop.shape, (2*r+1, 2*r+1))


class test_find_background_value(unittest.TestCase):

    def test_percentile(self):
        data = np.arange(100)
        value = _find_background_value(data, lowest_percentile=0.01)
        self.assertEqual(value, 0.)
        value = _find_background_value(data, lowest_percentile=0.1)
        self.assertEqual(value, 4.5)
        value = _find_background_value(data, lowest_percentile=0.5)
        self.assertEqual(value, 24.5)


class test_find_median_upper_percentile(unittest.TestCase):

    def test_percentile(self):
        data = np.arange(100)
        value = _find_median_upper_percentile(data, upper_percentile=0.01)
        self.assertEqual(value, 99.)
        value = _find_median_upper_percentile(data, upper_percentile=0.1)
        self.assertEqual(value, 94.5)
        value = _find_median_upper_percentile(data, upper_percentile=0.5)
        self.assertEqual(value, 74.5)
