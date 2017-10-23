import os
import unittest
import numpy as np
from hyperspy.api import load
from atomap.atom_position import Atom_Position
from atomap.sublattice import Sublattice
from atomap.testing_tools import MakeTestData
from atomap.atom_finding_refining import (
        _make_mask_from_positions, _crop_mask_slice_indices,
        _find_background_value, _find_median_upper_percentile,
        _make_model_from_atom_list, _fit_atom_positions_with_gaussian_model,
        _atom_to_gaussian_component, _make_circular_mask,
        fit_atom_positions_gaussian, subtract_average_background,
        do_pca_on_signal, get_atom_positions)

my_path = os.path.dirname(__file__)


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


class test_make_model_from_atom_list(unittest.TestCase):

    def setUp(self):
        image_data = np.random.random(size=(100, 100))
        position_list = []
        for x in range(10, 100, 5):
            for y in range(10, 100, 5):
                position_list.append([x, y])
        sublattice = Sublattice(np.array(position_list), image_data)
        sublattice.find_nearest_neighbors()
        self.sublattice = sublattice

    def test_1_atom(self):
        sublattice = self.sublattice
        model, mask = _make_model_from_atom_list(
                [sublattice.atom_list[10]],
                sublattice.image)
        self.assertEqual(len(model), 1)

    def test_2_atom(self):
        sublattice = self.sublattice
        model, mask = _make_model_from_atom_list(
                sublattice.atom_list[10:12],
                sublattice.image)
        self.assertEqual(len(model), 2)

    def test_5_atom(self):
        sublattice = self.sublattice
        model, mask = _make_model_from_atom_list(
                sublattice.atom_list[10:15],
                sublattice.image)
        self.assertEqual(len(model), 5)

    def test_set_mask_radius_atom(self):
        atom_list = [Atom_Position(2, 2), Atom_Position(4, 4)]
        image = np.random.random((20, 20))
        model, mask = _make_model_from_atom_list(
                atom_list=atom_list,
                image_data=image,
                mask_radius=3)
        self.assertEqual(len(model), 2)


class test_fit_atom_positions_with_gaussian_model(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(100, 100)
        x, y = np.mgrid[10:90:10j, 10:90:10j]
        x, y = x.flatten(), y.flatten()
        sigma, A = 1, 50
        test_data.add_atom_list(
                x, y, sigma_x=sigma, sigma_y=sigma, amplitude=A)
        self.sublattice = test_data.sublattice
        self.sublattice.find_nearest_neighbors()

    def test_1_atom(self):
        sublattice = self.sublattice
        g_list = _fit_atom_positions_with_gaussian_model(
                [sublattice.atom_list[5]],
                sublattice.image)
        self.assertEqual(len(g_list), 1)

    def test_2_atom(self):
        sublattice = self.sublattice
        g_list = _fit_atom_positions_with_gaussian_model(
                sublattice.atom_list[5:7],
                sublattice.image)
        self.assertEqual(len(g_list), 2)

    def test_5_atom(self):
        sublattice = self.sublattice
        g_list = _fit_atom_positions_with_gaussian_model(
                sublattice.atom_list[5:10],
                sublattice.image)
        self.assertEqual(len(g_list), 5)

    @unittest.expectedFailure
    def test_wrong_input_0(self):
        sublattice = self.sublattice
        _fit_atom_positions_with_gaussian_model(
                sublattice.atom_list[5],
                sublattice.image)

    @unittest.expectedFailure
    def test_wrong_input_1(self):
        sublattice = self.sublattice
        _fit_atom_positions_with_gaussian_model(
                [sublattice.atom_list[5:7]],
                sublattice.image)


class test_atom_to_gaussian_component(unittest.TestCase):

    def test_simple(self):
        x, y, sX, sY, r = 7.1, 2.8, 2.1, 3.3, 1.9
        atom_position = Atom_Position(
                x=x, y=y,
                sigma_x=sX, sigma_y=sY,
                rotation=r)
        gaussian = _atom_to_gaussian_component(atom_position)
        self.assertEqual(x, gaussian.centre_x.value)
        self.assertEqual(y, gaussian.centre_y.value)
        self.assertEqual(sX, gaussian.sigma_x.value)
        self.assertEqual(sY, gaussian.sigma_y.value)
        self.assertEqual(r, gaussian.rotation.value)


class test_make_circular_mask(unittest.TestCase):

    def test_small_radius_1(self):
        imX, imY = 3, 3
        mask = _make_circular_mask(1, 1, imX, imY, 1)
        self.assertEqual(mask.size, imX*imY)
        self.assertEqual(mask.sum(), 5)
        true_index = [[1, 0], [0, 1], [1, 1],  [2, 1], [1, 2]]
        false_index = [[0, 0], [0, 2], [2, 0],  [2, 2]]
        for index in true_index:
            self.assertTrue(mask[index[0], index[1]])
        for index in false_index:
            self.assertFalse(mask[index[0], index[1]])

    def test_all_true_mask(self):
        imX, imY = 5, 5
        mask = _make_circular_mask(1, 1, imX, imY, 5)
        self.assertTrue(mask.all())
        self.assertEqual(mask.size, imX*imY)
        self.assertEqual(mask.sum(), imX*imY)

    def test_all_false_mask(self):
        mask = _make_circular_mask(10, 10, 5, 5, 3)
        self.assertFalse(mask.any())


class test_fit_atom_positions_gaussian(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(100, 100)
        x, y = np.mgrid[5:95:10j, 5:95:10j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        sublattice = test_data.sublattice
        sublattice.construct_zone_axes()
        self.sublattice = sublattice
        self.x, self.y = x, y

    def test_one_atoms(self):
        sublattice = self.sublattice
        atom_index = 55
        atom_list = [sublattice.atom_list[atom_index]]
        image_data = sublattice.image
        fit_atom_positions_gaussian(atom_list, image_data)
        self.assertAlmostEqual(
                sublattice.atom_list[atom_index].pixel_x,
                self.x[atom_index], places=4)
        self.assertAlmostEqual(
                sublattice.atom_list[atom_index].pixel_y,
                self.y[atom_index], places=4)

    def test_two_atoms(self):
        sublattice = self.sublattice
        atom_indices = [44, 45]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            self.assertAlmostEqual(
                    sublattice.atom_list[atom_index].pixel_x,
                    self.x[atom_index], places=4)
            self.assertAlmostEqual(
                    sublattice.atom_list[atom_index].pixel_y,
                    self.y[atom_index], places=4)

    def test_four_atoms(self):
        sublattice = self.sublattice
        atom_indices = [35, 36, 45, 46]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            self.assertAlmostEqual(
                    sublattice.atom_list[atom_index].pixel_x,
                    self.x[atom_index], places=4)
            self.assertAlmostEqual(
                    sublattice.atom_list[atom_index].pixel_y,
                    self.y[atom_index], places=4)

    def test_nine_atoms(self):
        sublattice = self.sublattice
        atom_indices = [34, 35, 36, 44, 45, 46, 54, 55, 56]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            self.assertAlmostEqual(
                    sublattice.atom_list[atom_index].pixel_x,
                    self.x[atom_index], places=4)
            self.assertAlmostEqual(
                    sublattice.atom_list[atom_index].pixel_y,
                    self.y[atom_index], places=4)


class test_get_atom_positions(unittest.TestCase):

    def setUp(self):
        s_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        self.s_adf_modified = do_pca_on_signal(s_adf_modified)
        self.pixel_separation = peak_separation/s_adf.axes_manager[0].scale

    def test_find_number_of_columns(self):
        peaks = get_atom_positions(
                self.s_adf_modified,
                self.pixel_separation)
        self.assertEqual(len(peaks), 238)


class test_bad_fit_condition(unittest.TestCase):

    def setUp(self):
        t = MakeTestData(40, 40)
        x, y = np.mgrid[5:40:10, 5:40:10]
        x, y = x.flatten(), y.flatten()
        t.add_atom_list(x, y)
        self.sublattice = t.sublattice
        self.sublattice.find_nearest_neighbors()

    def test_initial_position_inside_mask_x(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        x0 = atom[0].pixel_x
        atom[0].pixel_x += 2
        g = _fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=4)
        self.assertAlmostEqual(g[0].centre_x.value, x0, places=1)

    def test_initial_position_outside_mask_x(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_x += 3
        g = _fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=2)
        self.assertFalse(g)

    def test_initial_position_outside_mask_y(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_y -= 4
        g = _fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=2)
        self.assertFalse(g)

    def test_initial_position_outside_mask_xy(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_y += 3
        atom[0].pixel_x += 3
        g = _fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=2)
        self.assertFalse(g)
