import matplotlib
matplotlib.use('Agg')
import os
import unittest
import numpy as np
import atomap.api as am
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        construct_zone_axes_from_sublattice,\
        get_atom_positions
from atomap.sublattice import Sublattice
from hyperspy.api import load
from atomap.atom_finding_refining import refine_sublattice
from atomap.atom_lattice import Atom_Lattice
import atomap.testing_tools as tt
from atomap.testing_tools import make_artifical_atomic_signal

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
        sublattice = Sublattice(
                self.peaks,
                np.rot90(np.fliplr(self.s_adf_modified.data)))

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
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
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
        s = self.sublattice.get_atom_distance_difference_map(
                [zone_vector])

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
        sublattice.get_atom_distance_difference_line_profile(zone_vector, plane)

    def test_get_atom_distance_difference_map(self):
        self.sublattice.get_atom_distance_difference_map()

    def test_get_atom_distance_map(self):
        self.sublattice.get_atom_distance_map()

    def test_zone_vector_mean_angle(self):
        zone_vector = self.sublattice.zones_axis_average_distances[0]
        mean_angle = self.sublattice.get_zone_vector_mean_angle(zone_vector)

    def test_get_nearest_neighbor_directions(self):
        self.sublattice.get_nearest_neighbor_directions()

    def test_get_property_line_profile(self):
        plane = self.sublattice.atom_plane_list[3]
        self.sublattice._get_property_line_profile(
                self.sublattice.x_position,
                self.sublattice.y_position,
                self.sublattice.ellipticity,
                atom_plane=plane)

    def test_get_ellipticity_vector(self):
        sublattice = self.sublattice
        s0 = sublattice.get_ellipticity_vector(
                color='blue',
                vector_scale=20)
        self.assertEqual(len(sublattice.atom_list), len(s0.metadata.Markers))
        plane = sublattice.atom_plane_list[3]
        s1 = sublattice.get_ellipticity_vector(
                sublattice.image,
                atom_plane_list=[plane],
                color='red',
                vector_scale=10)


class test_sublattice_interpolation(unittest.TestCase):

    def setUp(self):
        x, y = np.mgrid[0:100,0:100]
        position_list = np.vstack((x.flatten(), y.flatten())).T
        image_data = np.arange(100).reshape(10,10)
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
        x, y = np.mgrid[0:500:20j,0:500:20j]
        x, y = x.flatten(), y.flatten()
        s, g_list = tt.make_artifical_atomic_signal(x, y, image_pad=10)

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
        for x in range(10, 100, 5):
            for y in range(10, 100, 5):
                position_list.append([x, y])
        sublattice = Sublattice(np.array(position_list), image_data)
        self.sublattice = sublattice

    def test_simple(self):
        sublattice = self.sublattice
        sublattice.get_model_image()

class test_get_position_history(unittest.TestCase):

    def setUp(self):
        pos = [[x, y] for x in range(9) for y in range(9)]
        self.sublattice = Sublattice(pos, np.random.random((9, 9)))

    def test_no_history(self):
        sublattice = self.sublattice
        s = sublattice.get_position_history()

    def test_1_history(self):
        sublattice = self.sublattice
        for atom in sublattice.atom_list:
            atom.old_pixel_x_list.append(np.random.random())
            atom.old_pixel_y_list.append(np.random.random())
        s = sublattice.get_position_history()


class test_get_atom_angles_from_zone_vector(unittest.TestCase):
    def setUp(self):
        x, y = np.mgrid[0:500:10j,0:500:10j]
        x, y = x.flatten(), y.flatten()
        sigma_value = 10
        sigma = [sigma_value]*len(x)
        A = [50]*len(x)
        s, g_list = make_artifical_atomic_signal(
                x, y, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=100)
        atom_lattice = am.make_atom_lattice_from_image(
                s,
                am.process_parameters.GenericStructure(),
                pixel_separation=40)
        self.sublattice = atom_lattice.sublattice_list[0]

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
        x, y = np.mgrid[0:500:10j,0:500:10j]
        x, y = x.flatten(), y.flatten()
        sigma_value = 10
        sigma = [sigma_value]*len(x)
        A = [50]*len(x)
        s, g_list = make_artifical_atomic_signal(
                x, y, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=100)
        atom_lattice = am.make_atom_lattice_from_image(
                s,
                am.process_parameters.GenericStructure(),
                pixel_separation=40)
        self.sublattice = atom_lattice.sublattice_list[0]

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
        x, y = np.mgrid[0:500:8j,0:500:8j]
        x, y = x.flatten(), y.flatten()
        sigma_value = 10
        sigma = [sigma_value]*len(x)
        image_pad = 20
        A = [50]*len(x)
        s, g_list = make_artifical_atomic_signal(
                x, y, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=image_pad)
        self.image_data = s.data
        self.xy = np.dstack((x + image_pad, y + image_pad))[0]

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
        x, y = np.mgrid[0:500:10j,0:500:10j]
        x, y = x.flatten(), y.flatten()
        sigma_value = 10
        sigma = [sigma_value]*len(x)
        A = [50]*len(x)
        s, g_list = make_artifical_atomic_signal(
                x, y, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=100)
        atom_lattice = am.make_atom_lattice_from_image(
                s,
                am.process_parameters.GenericStructure(),
                pixel_separation=40)
        self.sublattice = atom_lattice.sublattice_list[0]

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
