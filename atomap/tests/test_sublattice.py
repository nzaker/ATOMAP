import os
import unittest
import numpy as np
from hyperspy.api import load
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        construct_zone_axes_from_sublattice,\
        get_atom_positions
from atomap.sublattice import Sublattice
from atomap.atom_finding_refining import refine_sublattice
import atomap.testing_tools as tt
import atomap.dummy_data as dd
from atomap.testing_tools import MakeTestData

my_path = os.path.dirname(__file__)


class test_make_simple_sublattice(unittest.TestCase):
    def setUp(self):
        self.atoms_N = 10
        self.image_data = np.arange(10000).reshape(100, 100)
        self.peaks = np.arange(self.atoms_N*2).reshape(self.atoms_N, 2)

    def test_make_sublattice(self):
        sublattice = Sublattice(
                self.peaks,
                self.image_data)
        self.assertEqual(len(sublattice.atom_list), self.atoms_N)

    def test_repr_no_planes(self):
        sublattice = Sublattice(
                self.peaks,
                self.image_data)
        sublattice.name = 'test'
        repr_str = '<Sublattice, test (atoms:%s,planes:0)>' % (
                self.atoms_N)
        self.assertEqual(
                sublattice.__repr__(),
                repr_str)


class test_sublattice_with_atom_planes(unittest.TestCase):

    def setUp(self):
        s_adf_filename = os.path.join(
                my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        self.s_adf_modified = do_pca_on_signal(s_adf_modified)
        self.pixel_size = s_adf.axes_manager[0].scale
        self.pixel_separation = peak_separation/self.pixel_size

        self.peaks = get_atom_positions(
                self.s_adf_modified,
                self.pixel_separation)

    def test_make_sublattice(self):
        Sublattice(self.peaks, np.rot90(np.fliplr(self.s_adf_modified.data)))

    def test_make_construct_zone_axes(self):
        sublattice = Sublattice(
                self.peaks,
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        sublattice.pixel_size = self.pixel_size
        construct_zone_axes_from_sublattice(sublattice)

        number_zone_vector_110 = len(
                sublattice.atom_planes_by_zone_vector[
                    sublattice.zones_axis_average_distances[0]])
        number_zone_vector_100 = len(
                sublattice.atom_planes_by_zone_vector[
                    sublattice.zones_axis_average_distances[1]])

        self.assertEqual(number_zone_vector_110, 14)
        self.assertEqual(number_zone_vector_100, 17)

    def test_repr(self):
        sublattice = Sublattice(
                self.peaks,
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        sublattice.pixel_size = self.pixel_size
        sublattice.name = 'test planes'
        construct_zone_axes_from_sublattice(sublattice)

        repr_str = '<Sublattice, test planes (atoms:%s,planes:%s)>' % (
                len(sublattice.atom_list),
                len(sublattice.atom_planes_by_zone_vector))
        self.assertEqual(
                sublattice.__repr__(),
                repr_str)

    def test_get_zone_vector_index(self):
        sublattice = Sublattice(
                self.peaks,
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        sublattice.pixel_size = self.pixel_size
        construct_zone_axes_from_sublattice(sublattice)
        zone_axis_index = sublattice.get_zone_vector_index(
                sublattice.zones_axis_average_distances_names[0])
        self.assertEqual(zone_axis_index, 0)
        with self.assertRaises(ValueError):
            sublattice.get_zone_vector_index('(99, 99)')

    def test_center_of_mass_refine(self):
        sublattice = Sublattice(
                self.peaks,
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        sublattice.pixel_size = self.pixel_size
        refine_sublattice(
                sublattice,
                [
                    (sublattice.image, 1, 'center_of_mass')],
                0.25)


class test_sublattice_get_signal(unittest.TestCase):

    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets",
                                      "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_atom_positions(
                s_adf_modified,
                pixel_separation)
        self.sublattice = Sublattice(
                peaks,
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sublattice.original_image = np.rot90(np.fliplr(s_adf.data))
        self.sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(self.sublattice)

    def test_ellipticity_map(self):
        self.sublattice.get_ellipticity_map()

    def test_ellipticity_line_line_profile(self):
        plane = self.sublattice.atom_plane_list[3]
        self.sublattice.get_ellipticity_line_profile(plane)

    def test_distance_difference(self):
        zone_vector = self.sublattice.zones_axis_average_distances[0]
        self.sublattice.get_atom_distance_difference_map([zone_vector])

    def test_atomplanes_from_zone_vector(self):
        sublattice = self.sublattice
        s_list = sublattice.get_all_atom_planes_by_zone_vector()
        self.assertEqual(
                len(s_list),
                len(sublattice.zones_axis_average_distances))

    def test_atomap_plane_on_image(self):
        sublattice = self.sublattice
        atom_planes = sublattice.atom_plane_list[10:20]
        number_of_atom_planes = 0
        for atom_plane in atom_planes:
            number_of_atom_planes += len(atom_plane.atom_list) - 1
        s = sublattice.get_atom_planes_on_image(
                atom_planes, add_numbers=False)
        self.assertEqual(number_of_atom_planes, len(s.metadata.Markers))

    def test_get_atom_list(self):
        self.sublattice.get_atom_list_on_image()

    def test_get_monolayer_distance_line_profile(self):
        sublattice = self.sublattice
        zone_vector = sublattice.zones_axis_average_distances[0]
        plane = sublattice.atom_planes_by_zone_vector[zone_vector][0]
        sublattice.get_monolayer_distance_line_profile(zone_vector, plane)

    def test_get_monolayer_distance_map(self):
        self.sublattice.get_monolayer_distance_map()

    def test_get_atom_distance_difference_line_profile(self):
        sublattice = self.sublattice
        zone_vector = sublattice.zones_axis_average_distances[0]
        plane = sublattice.atom_planes_by_zone_vector[zone_vector][0]
        sublattice.get_atom_distance_difference_line_profile(
                    zone_vector, plane)

    def test_get_atom_distance_difference_map(self):
        self.sublattice.get_atom_distance_difference_map()

    def test_get_atom_distance_map(self):
        self.sublattice.get_atom_distance_map()

    def test_zone_vector_mean_angle(self):
        zone_vector = self.sublattice.zones_axis_average_distances[0]
        self.sublattice.get_zone_vector_mean_angle(zone_vector)

    def test_get_nearest_neighbor_directions(self):
        self.sublattice.get_nearest_neighbor_directions()

    def test_get_ellipticity_vector(self):
        sublattice = self.sublattice
        s0 = sublattice.get_ellipticity_vector(
                color='blue',
                vector_scale=20)
        self.assertEqual(len(sublattice.atom_list), len(s0.metadata.Markers))
        plane = sublattice.atom_plane_list[3]
        sublattice.get_ellipticity_vector(
                sublattice.image,
                atom_plane_list=[plane],
                color='red',
                vector_scale=10)


class test_sublattice_interpolation(unittest.TestCase):

    def setUp(self):
        x, y = np.mgrid[0:100, 0:100]
        position_list = np.vstack((x.flatten(), y.flatten())).T
        image_data = np.arange(100).reshape(10, 10)
        self.sublattice = Sublattice(position_list, image_data)

    def test_make_sublattice(self):
        sublattice = self.sublattice
        x_list = sublattice.x_position
        y_list = sublattice.y_position
        z_list = np.zeros_like(x_list)
        output = sublattice._get_regular_grid_from_unregular_property(
                x_list, y_list, z_list)
        z_interpolate = output[2]
        self.assertTrue(not z_interpolate.any())


class test_sublattice_fingerprinter(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(520, 520)
        x, y = np.mgrid[10:510:20j, 10:510:20j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        s = test_data.signal

        atom_positions = get_atom_positions(
                signal=s,
                separation=10,
                threshold_rel=0.02,
                )
        sublattice = Sublattice(
                atom_position_list=atom_positions,
                image=s.data)
        sublattice.find_nearest_neighbors()
        self.sublattice = sublattice

    def test_fingerprint_1d(self):
        sublattice = self.sublattice
        sublattice.get_fingerprint_1d()

    def test_fingerprint_2d(self):
        sublattice = self.sublattice
        sublattice.get_fingerprint_2d()

    def test_fingerprint(self):
        sublattice = self.sublattice
        sublattice._get_fingerprint()


class test_sublattice_get_model_image(unittest.TestCase):

    def setUp(self):
        image_data = np.random.random(size=(100, 100))
        position_list = []
        for x in range(5, 100, 5):
            for y in range(5, 100, 5):
                position_list.append([x, y])
        sublattice = Sublattice(np.array(position_list), image_data)
        self.sublattice = sublattice

    def test_simple(self):
        sublattice = self.sublattice
        sublattice.get_model_image()

    def test_image_shape_default(self):
        sublattice = self.sublattice
        s = sublattice.get_model_image()
        self.assertEqual(s.axes_manager.shape, sublattice.image.shape)

    def test_image_shape_small(self):
        sublattice = self.sublattice
        s = sublattice.get_model_image(image_shape=(10, 10))
        self.assertEqual(s.axes_manager.shape, (10, 10))

    def test_image_shape_large(self):
        sublattice = self.sublattice
        s = sublattice.get_model_image(image_shape=(1000, 1000))
        self.assertEqual(s.axes_manager.shape, (1000, 1000))

    def test_image_shape_rectangle_1(self):
        sublattice = self.sublattice
        s = sublattice.get_model_image(image_shape=(100, 200))
        self.assertEqual(s.axes_manager.shape, (100, 200))

    def test_image_shape_rectangle_2(self):
        sublattice = self.sublattice
        s = sublattice.get_model_image(image_shape=(200, 100))
        self.assertEqual(s.axes_manager.shape, (200, 100))


class test_get_position_history(unittest.TestCase):

    def setUp(self):
        pos = [[x, y] for x in range(9) for y in range(9)]
        self.sublattice = Sublattice(pos, np.random.random((9, 9)))

    def test_no_history(self):
        sublattice = self.sublattice
        sublattice.get_position_history()

    def test_1_history(self):
        sublattice = self.sublattice
        for atom in sublattice.atom_list:
            atom.old_pixel_x_list.append(np.random.random())
            atom.old_pixel_y_list.append(np.random.random())
        sublattice.get_position_history()


class test_get_atom_angles_from_zone_vector(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(700, 700)
        x, y = np.mgrid[100:600:10j, 100:600:10j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y, sigma_x=10, sigma_y=10, amplitude=50)
        self.sublattice = test_data.sublattice
        self.sublattice.construct_zone_axes()

    def test_cubic_radians(self):
        sublattice = self.sublattice
        x, y, z = sublattice.get_atom_angles_from_zone_vector(
                sublattice.zones_axis_average_distances,
                sublattice.zones_axis_average_distances)
        self.assertTrue(
                np.allclose(np.zeros_like(z)+np.pi/2, z))

    def test_cubic_degrees(self):
        sublattice = self.sublattice
        x, y, z = sublattice.get_atom_angles_from_zone_vector(
                sublattice.zones_axis_average_distances,
                sublattice.zones_axis_average_distances,
                degrees=True)
        self.assertTrue(
                np.allclose(np.zeros_like(z)+90, z))


class test_get_atom_plane_slice_between_two_planes(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(700, 700)
        x, y = np.mgrid[100:600:10j, 100:600:10j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y, sigma_x=10, sigma_y=10, amplitude=50)
        self.sublattice = test_data.sublattice
        self.sublattice.construct_zone_axes()

    def test_subset0(self):
        sublattice = self.sublattice
        zv = sublattice.zones_axis_average_distances[0]
        ap_index0, ap_index1 = 2, 6
        ap0 = sublattice.atom_planes_by_zone_vector[zv][ap_index0]
        ap1 = sublattice.atom_planes_by_zone_vector[zv][ap_index1]
        ap_list = sublattice.get_atom_plane_slice_between_two_planes(
                ap0, ap1)
        self.assertEqual(ap_index1-ap_index0+1, len(ap_list))
        ap_list_check = sublattice.atom_planes_by_zone_vector[
                zv][ap_index0:ap_index1+1]
        self.assertTrue(ap_list, ap_list_check)

    def test_subset1(self):
        sublattice = self.sublattice
        zv = sublattice.zones_axis_average_distances[0]
        ap_index0, ap_index1 = 3, 4
        ap0 = sublattice.atom_planes_by_zone_vector[zv][ap_index0]
        ap1 = sublattice.atom_planes_by_zone_vector[zv][ap_index1]
        ap_list = sublattice.get_atom_plane_slice_between_two_planes(
                ap0, ap1)
        self.assertEqual(ap_index1-ap_index0+1, len(ap_list))
        ap_list_check = sublattice.atom_planes_by_zone_vector[
                zv][ap_index0:ap_index1+1]
        self.assertTrue(ap_list, ap_list_check)

    def test_get_all(self):
        sublattice = self.sublattice
        zv = sublattice.zones_axis_average_distances[0]
        ap_index0, ap_index1 = 0, -1
        ap0 = sublattice.atom_planes_by_zone_vector[zv][ap_index0]
        ap1 = sublattice.atom_planes_by_zone_vector[zv][ap_index1]
        ap_list = sublattice.get_atom_plane_slice_between_two_planes(
                ap0, ap1)
        self.assertEqual(10, len(ap_list))
        ap_list_check = sublattice.atom_planes_by_zone_vector[
                zv]
        self.assertTrue(ap_list, ap_list_check)


class test_refine_functions(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(540, 540)
        x, y = np.mgrid[20:520:8j, 20:520:8j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y, sigma_x=10, sigma_y=10, amplitude=50)
        self.image_data = test_data.signal.data
        self.xy = np.dstack((x, y))[0]

    def test_refine_2d_gaussian_simple(self):
        sublattice = Sublattice(self.xy, self.image_data)
        with self.assertRaises(ValueError):
            sublattice.refine_atom_positions_using_2d_gaussian()
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_2d_gaussian()

    def test_refine_2d_gaussian_all_arguments(self):
        sublattice = Sublattice(self.xy, self.image_data)
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_2d_gaussian(
                image_data=self.image_data, percent_to_nn=0.3,
                rotation_enabled=False)

    def test_refine_center_of_mass_simple(self):
        sublattice = Sublattice(self.xy, self.image_data)
        with self.assertRaises(ValueError):
            sublattice.refine_atom_positions_using_center_of_mass()
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_center_of_mass()

    def test_refine_center_of_mass_all_arguments(self):
        sublattice = Sublattice(self.xy, self.image_data)
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_center_of_mass(
                image_data=self.image_data, percent_to_nn=0.3)


class test_get_atom_list_between_four_atom_planes(unittest.TestCase):

    def setUp(self):
        test_data = MakeTestData(700, 700)
        x, y = np.mgrid[100:600:10j, 100:600:10j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y, sigma_x=10, sigma_y=10, amplitude=50)
        self.sublattice = test_data.sublattice
        self.sublattice.construct_zone_axes()

    def test_subset0(self):
        sublattice = self.sublattice
        zv0 = sublattice.zones_axis_average_distances[0]
        zv1 = sublattice.zones_axis_average_distances[1]
        ap_index00, ap_index01 = 2, 6
        ap_index10, ap_index11 = 2, 6
        ap00 = sublattice.atom_planes_by_zone_vector[zv0][ap_index00]
        ap01 = sublattice.atom_planes_by_zone_vector[zv0][ap_index01]
        ap10 = sublattice.atom_planes_by_zone_vector[zv1][ap_index10]
        ap11 = sublattice.atom_planes_by_zone_vector[zv1][ap_index11]
        apos_list = sublattice.get_atom_list_between_four_atom_planes(
                ap00, ap01, ap10, ap11)

        num_atoms = (ap_index01-ap_index00+1)*(ap_index11-ap_index10+1)
        self.assertEqual(num_atoms, len(apos_list))

    def test_subset1(self):
        sublattice = self.sublattice
        zv0 = sublattice.zones_axis_average_distances[0]
        zv1 = sublattice.zones_axis_average_distances[1]
        ap_index00, ap_index01 = 5, 8
        ap_index10, ap_index11 = 2, 5
        ap00 = sublattice.atom_planes_by_zone_vector[zv0][ap_index00]
        ap01 = sublattice.atom_planes_by_zone_vector[zv0][ap_index01]
        ap10 = sublattice.atom_planes_by_zone_vector[zv1][ap_index10]
        ap11 = sublattice.atom_planes_by_zone_vector[zv1][ap_index11]
        apos_list = sublattice.get_atom_list_between_four_atom_planes(
                ap00, ap01, ap10, ap11)

        num_atoms = (ap_index01-ap_index00+1)*(ap_index11-ap_index10+1)
        self.assertEqual(num_atoms, len(apos_list))

    def test_get_all(self):
        sublattice = self.sublattice
        zv0 = sublattice.zones_axis_average_distances[0]
        zv1 = sublattice.zones_axis_average_distances[1]
        ap_index00, ap_index01 = 0, -1
        ap_index10, ap_index11 = 0, -1
        ap00 = sublattice.atom_planes_by_zone_vector[zv0][ap_index00]
        ap01 = sublattice.atom_planes_by_zone_vector[zv0][ap_index01]
        ap10 = sublattice.atom_planes_by_zone_vector[zv1][ap_index10]
        ap11 = sublattice.atom_planes_by_zone_vector[zv1][ap_index11]
        apos_list = sublattice.get_atom_list_between_four_atom_planes(
                ap00, ap01, ap10, ap11)

        num_atoms = 100
        self.assertEqual(num_atoms, len(apos_list))


class test_make_translation_symmetry(unittest.TestCase):

    def test_cubic_simple(self):
        vX, vY = 10, 10
        x, y = np.mgrid[5:95:vX, 5:95:vY]
        test_data = MakeTestData(100, 100)
        test_data.add_atom_list(x.flatten(), y.flatten(), sigma_x=2, sigma_y=2)
        sublattice = test_data.sublattice
        sublattice._pixel_separation = sublattice._get_pixel_separation()
        sublattice._make_translation_symmetry()
        zone_vectors = sublattice.zones_axis_average_distances
        self.assertEqual(zone_vectors[0], (0, vY))
        self.assertEqual(zone_vectors[1], (vX, 0))
        self.assertEqual(zone_vectors[2], (vX, vY))
        self.assertEqual(zone_vectors[3], (vX, -vY))
        self.assertEqual(zone_vectors[4], (vX, 2*vY))
        self.assertEqual(zone_vectors[5], (2*vX, vY))

    def test_rectangle_simple(self):
        vX, vY = 10, 15
        x, y = np.mgrid[5:95:vX, 5:95:vY]
        test_data = MakeTestData(100, 100)
        test_data.add_atom_list(x.flatten(), y.flatten(), sigma_x=2, sigma_y=2)
        sublattice = test_data.sublattice
        sublattice._pixel_separation = sublattice._get_pixel_separation()
        sublattice._make_translation_symmetry()
        zone_vectors = sublattice.zones_axis_average_distances
        self.assertEqual(zone_vectors[0], (vX, 0))
        self.assertEqual(zone_vectors[1], (0, vY))
        self.assertEqual(zone_vectors[2], (vX, vY))
        self.assertEqual(zone_vectors[3], (-vX, vY))
        self.assertEqual(zone_vectors[4], (2*vX, vY))
        self.assertEqual(zone_vectors[5], (2*vX, -vY))


class test_construct_zone_axes(unittest.TestCase):

    def test_cubic_simple(self):
        vX, vY = 10, 10
        x, y = np.mgrid[5:95:vX, 5:95:vY]
        test_data = MakeTestData(100, 100)
        test_data.add_atom_list(x.flatten(), y.flatten(), sigma_x=2, sigma_y=2)
        sublattice = test_data.sublattice
        sublattice.construct_zone_axes()
        zone_vectors = sublattice.zones_axis_average_distances
        self.assertEqual(zone_vectors[0], (0, vY))
        self.assertEqual(zone_vectors[1], (vX, 0))
        self.assertEqual(zone_vectors[2], (vX, vY))
        self.assertEqual(zone_vectors[3], (vX, -vY))

    def test_rectangle(self):
        vX, vY = 15, 10
        x, y = np.mgrid[5:95:vX, 5:95:vY]
        test_data = MakeTestData(100, 100)
        test_data.add_atom_list(x.flatten(), y.flatten(), sigma_x=2, sigma_y=2)
        sublattice = test_data.sublattice
        sublattice.construct_zone_axes()
        zone_vectors = sublattice.zones_axis_average_distances
        self.assertEqual(zone_vectors[0], (0, vY))
        self.assertEqual(zone_vectors[1], (vX, 0))
        self.assertEqual(zone_vectors[2], (vX, vY))
        self.assertEqual(zone_vectors[3], (vX, -vY))
        self.assertEqual(zone_vectors[4], (vX, 2*vY))
        self.assertEqual(zone_vectors[5], (-vX, 2*vY))


class test_sublattice_mask(unittest.TestCase):

    def setUp(self):
        image_data = np.random.random(size=(100, 100))
        position_list = []
        for x in range(10, 100, 10):
            for y in range(10, 100, 10):
                position_list.append([x, y])
        sublattice = Sublattice(np.array(position_list), image_data)
        self.sublattice = sublattice

    def test_radius_is_0(self):
        sublattice = self.sublattice
        s = sublattice.mask_image_around_sublattice(
            image_data=sublattice.image, radius=0)
        self.assertEqual(np.count_nonzero(s.data), len(sublattice.atom_list))


class test_plot_functions(unittest.TestCase):

    def setUp(self):
        self.sublattice = dd.get_simple_cubic_sublattice()

    def test_plot(self):
        self.sublattice.plot()
        self.sublattice.plot(color='green', cmap='viridis')

    def test_plot_planes(self):
        self.sublattice.construct_zone_axes()
        self.sublattice.plot_planes(
                        color='green',
                        add_numbers=True,
                        cmap='viridis')

    def test_plot_ellipticity_vectors(self):
        self.sublattice.plot_ellipticity_vectors(save=True)

    def test_plot_ellipticity_map(self):
        self.sublattice.plot_ellipticity_map(cmap='viridis')


class test_mask_indices(unittest.TestCase):

    def setUp(self):
        t1 = tt.MakeTestData(20, 10)
        t1.add_atom_list([5, 15], [5, 5])
        sublattice = t1.sublattice
        self.mask_list = sublattice._get_sublattice_atom_list_mask()
        s = sublattice.mask_image_around_sublattice(sublattice.image, radius=1)
        self.s_i = np.asarray(np.nonzero(s.data))

    def test_mask_atom_list_len(self):
        mask_list = self.mask_list
        self.assertTrue(len(mask_list) == 2)

    def test_mask_image_len(self):
        s_i_size = self.s_i[0].size
        b_size = self.mask_list[1][0].size
        a_size = self.mask_list[0][0].size
        self.assertTrue(s_i_size == (b_size + a_size))

    def test_indices_equal(self):
        mask_list = self.mask_list
        a = np.asarray(mask_list[0]).T
        b = np.asarray(mask_list[1]).T
        j = 0
        s_i = self.s_i
        for x in s_i[0]:
            y = s_i[1][j]
            j += 1
            A = ((np.array([x, y]) == a).all(1).any())
            B = ((np.array([x, y]) == b).all(1).any())
            self.assertTrue(A or B)


class test_get_property_line_profile(unittest.TestCase):

    def setUp(self):
        x, y = np.mgrid[5:50:5, 5:50:5]
        x, y = x.flatten(), y.flatten()

        tV = tt.MakeTestData(50, 50)
        tV.add_atom_list(x, y)
        sublatticeV = tV.sublattice
        sublatticeV.construct_zone_axes()
        self.sublatticeV = sublatticeV

        tH = tt.MakeTestData(50, 50)
        tH.add_atom_list(y, x)
        sublatticeH = tH.sublattice
        sublatticeH.construct_zone_axes()
        self.sublatticeH = sublatticeH

        z_list = np.full_like(sublatticeV.x_position, 0)
        z_list[36:] = 1
        self.property_list = z_list

    def test_vertical_interface_horizontal_projection_plane(self):
        sublattice = self.sublatticeV
        property_list = self.property_list
        zone = sublattice.zones_axis_average_distances[0]
        plane = sublattice.atom_planes_by_zone_vector[zone][4]
        s = sublattice._get_property_line_profile(
                        sublattice.x_position,
                        sublattice.y_position,
                        property_list,
                        plane)
        self.assertAlmostEqual(s.isig[:-5.].data.all(), 0, places=2)
        self.assertAlmostEqual(s.isig[0.:].data.all(), 1, places=2)
        self.assertTrue(len(s.metadata['Markers'].keys()) == 9)

    def test_vertical_interface_vertical_projection_plane(self):
        sublattice = self.sublatticeV
        property_list = self.property_list
        zone = sublattice.zones_axis_average_distances[1]
        plane = sublattice.atom_planes_by_zone_vector[zone][0]
        s = sublattice._get_property_line_profile(
                        sublattice.x_position,
                        sublattice.y_position,
                        property_list,
                        plane)
        self.assertTrue((s.data == (5./9)).all())
        self.assertTrue(len(s.metadata['Markers'].keys()) == 9)

    def test_horizontal_interface_vertical_projection_plane(self):
        sublattice = self.sublatticeH
        property_list = self.property_list
        zone = sublattice.zones_axis_average_distances[0]
        plane = sublattice.atom_planes_by_zone_vector[zone][4]
        s = sublattice._get_property_line_profile(
                        sublattice.x_position,
                        sublattice.y_position,
                        property_list,
                        plane)
        self.assertAlmostEqual(s.isig[:0.].data.all(), 1, places=2)
        self.assertAlmostEqual(s.isig[5.:].data.all(), 0, places=2)
        self.assertTrue(len(s.metadata['Markers'].keys()) == 9)

    def test_horizontal_interface_horizontal_projection_plane(self):
        sublattice = self.sublatticeH
        property_list = self.property_list
        zone = sublattice.zones_axis_average_distances[1]
        plane = sublattice.atom_planes_by_zone_vector[zone][0]
        s = sublattice._get_property_line_profile(
                        sublattice.x_position,
                        sublattice.y_position,
                        property_list,
                        plane)
        self.assertTrue((s.data == (5./9)).all())
        self.assertTrue(len(s.metadata['Markers'].keys()) == 9)

    def test_metadata_line_profile_data(self):
        sublattice = self.sublatticeH
        zv = sublattice.zones_axis_average_distances[0]
        ap = sublattice.atom_planes_by_zone_vector[zv][4]
        data = sublattice.get_monolayer_distance_list_from_zone_vector(zv)
        s_l = sublattice._get_property_line_profile(
                data[0], data[1], data[2],
                atom_plane=ap)
        y_list = s_l.metadata.line_profile_data.y_list
        self.assertEqual(len(y_list), 8)
        np.testing.assert_allclose(y_list, np.ones_like(y_list)*5, atol=0.01)

    def test_wrong_input(self):
        sublattice = self.sublatticeH
        zv = sublattice.zones_axis_average_distances[0]
        ap = sublattice.atom_planes_by_zone_vector[zv][4]
        data = sublattice.get_monolayer_distance_list_from_zone_vector(zv)
        with self.assertRaises(ValueError):
            sublattice._get_property_line_profile(
                    data[0][:-2], data[1], data[2], atom_plane=ap)
        with self.assertRaises(ValueError):
            sublattice._get_property_line_profile(
                    data[0], data[1][:-3], data[2], atom_plane=ap)
        with self.assertRaises(ValueError):
            sublattice._get_property_line_profile(
                    data[0], data[1], data[2][:-1], atom_plane=ap)
        with self.assertRaises(ValueError):
            sublattice._get_property_line_profile(
                    data[0][:-3], data[1], data[2][:-3], atom_plane=ap)
        s_l = sublattice._get_property_line_profile(
                data[0][:-2], data[1][:-2], data[2][:-2], atom_plane=ap)
        s_l.plot()


class test_project_property_line_crossing(unittest.TestCase):

    def setUp(self):
        t = tt.MakeTestData(50, 50)
        x, y = np.mgrid[5:50:5, 5:50:5]
        x, y = x.flatten(), y.flatten()
        t.add_atom_list(x, y)
        sublattice = t.sublattice
        sublattice.construct_zone_axes()
        self.sublattice = sublattice

        x0, x1, idx = 1, 8, []
        for j in range(1, 9):
            span = np.arange(x0, x0+x1, 1)
            x0 += 10
            x1 -= 1
            idx.extend(span)

        idx = np.asarray(idx)

        z_list = np.full_like(sublattice.x_position, 0)
        z_list[idx] = 1
        self.property_list = z_list

    def test_projection_orhogonal(self):
        sublattice = self.sublattice
        property_list = self.property_list
        zone = sublattice.zones_axis_average_distances[2]
        plane = sublattice.atom_planes_by_zone_vector[zone][8]
        s = sublattice._get_property_line_profile(
                        sublattice.x_position,
                        sublattice.y_position,
                        property_list,
                        plane)
        self.assertAlmostEqual(s.isig[:0.].data.all(), 1, places=2)
        self.assertAlmostEqual(s.isig[5.:].data.all(), 0, places=2)


class test_get_property_map(unittest.TestCase):

    def setUp(self):
        t = tt.MakeTestData(30, 30)
        x, y = np.mgrid[5:30:5, 5:30:5]
        x, y = x.flatten(), y.flatten()
        t.add_atom_list(x, y)
        self.sublattice = t.sublattice
        self.z_list = np.full_like(t.sublattice.x_position, 1).tolist()

    def test_simple_map(self):
        sublattice = self.sublattice
        z_list = self.z_list
        s = sublattice.get_property_map(
                    sublattice.x_position,
                    sublattice.y_position,
                    z_list)
        self.assertTrue(s.axes_manager[0].scale == 0.5)
        self.assertTrue(s.axes_manager[1].scale == 0.5)
        self.assertTrue(s.data[10:50, 10:50].mean() == 1)

    def test_all_parameters(self):
        sublattice = self.sublattice
        sublattice.construct_zone_axes()
        z_list = self.z_list
        sub0 = Sublattice(np.array([[18, 15]]), image=sublattice.image)
        s = sublattice.get_property_map(
                    sublattice.x_position,
                    sublattice.y_position,
                    z_list,
                    atom_plane_list=[sublattice.atom_plane_list[0]],
                    add_zero_value_sublattice=sub0,
                    upscale_map=4
                    )
        self.assertTrue(s.axes_manager[0].scale == 0.25)
        self.assertTrue(s.axes_manager[1].scale == 0.25)
        self.assertTrue(s.data[20:100, 20:100].mean() <= 1)
        self.assertTrue(s.axes_manager[0].size == 120)
        self.assertTrue(s.axes_manager[1].size == 120)
        self.assertTrue(len(s.metadata['Markers'].keys()) == 4)
