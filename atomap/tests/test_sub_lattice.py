import os
import unittest
import numpy as np
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        construct_zone_axes_from_sub_lattice,\
        get_peak2d_skimage
from atomap.sub_lattice import Sub_Lattice
from hyperspy.api import load
from atomap.atom_finding_refining import refine_sub_lattice

from atomap.atom_lattice import Atom_Lattice

my_path = os.path.dirname(__file__)

class test_make_simple_sub_lattice(unittest.TestCase):
    def setUp(self):
        self.atoms_N = 10
        self.image_data = np.arange(10000).reshape(100,100)
        self.peaks = np.arange(20).reshape(self.atoms_N,2)

    def test_make_sub_lattice(self):
        sub_lattice = Sub_Lattice(
                self.peaks, 
                self.image_data)
        self.assertEqual(len(sub_lattice.atom_list), self.atoms_N)


class test_sub_lattice_construct_refine(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "/datasets/test_ADF_cropped.hdf5"
        peak_separation = 0.15

        s_adf = load(
                my_path +
                s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        self.s_adf_modified = do_pca_on_signal(s_adf_modified)
        self.pixel_size = s_adf.axes_manager[0].scale
        self.pixel_separation = peak_separation/self.pixel_size

        self.peaks = get_peak2d_skimage(
                self.s_adf_modified, 
                self.pixel_separation)[0]

    def test_make_sub_lattice(self):
        sub_lattice = Sub_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))

    def test_make_construct_zone_axes(self):
        sub_lattice = Sub_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        sub_lattice.pixel_size = self.pixel_size
        construct_zone_axes_from_sub_lattice(sub_lattice)

        number_zone_vector_110 = len(
                sub_lattice.atom_planes_by_zone_vector[
                    sub_lattice.zones_axis_average_distances[0]])
        number_zone_vector_100 = len(
                sub_lattice.atom_planes_by_zone_vector[
                    sub_lattice.zones_axis_average_distances[1]])

        self.assertEqual(number_zone_vector_110, 14)
        self.assertEqual(number_zone_vector_100, 17)

    def test_center_of_mass_refine(self):
        sub_lattice = Sub_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        sub_lattice.pixel_size = self.pixel_size
        refine_sub_lattice(
                sub_lattice, 
                [
                    (sub_lattice.adf_image, 1, 'center_of_mass')],
                0.25)
        sub_lattice.plot_atom_list_on_image_data(
                image=sub_lattice.adf_image,
                figname="com_test.jpg")


class test_sub_lattice_processing(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "/datasets/test_ADF_cropped.hdf5"
        peak_separation = 0.15

        s_adf = load(
                my_path +
                s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sub_lattice = Sub_Lattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sub_lattice.pixel_size = pixel_size
        construct_zone_axes_from_sub_lattice(self.sub_lattice)

    def test_zone_vector_mean_angle(self):
        zone_vector = self.sub_lattice.zones_axis_average_distances[0]
        mean_angle = self.sub_lattice.get_zone_vector_mean_angle(zone_vector)

    def test_get_nearest_neighbor_directions(self):
        self.sub_lattice.get_nearest_neighbor_directions()

class test_sub_lattice_plotting(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "/datasets/test_ADF_cropped.hdf5"
        peak_separation = 0.15

        s_adf = load(
                my_path +
                s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sub_lattice = Sub_Lattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sub_lattice.pixel_size = pixel_size
        construct_zone_axes_from_sub_lattice(self.sub_lattice)

    def test_plot_sub_lattice_atom_distance_map(self):
        self.sub_lattice.plot_atom_distance_map()

    def test_plot_sub_lattice_atom_distance_difference_map(self):
        self.sub_lattice.plot_atom_distance_difference_map()

    def test_plot_sub_lattice_monolayer_distance_map(self):
        self.sub_lattice.plot_monolayer_distance_map()

    def test_plot_sub_lattice_atom_list(self):
        self.sub_lattice.plot_atom_list_on_image_data(
                image=self.sub_lattice.adf_image)

    def test_plot_ellipticity_rotation_complex(self):
        self.sub_lattice.plot_ellipticity_rotation_complex()


class test_sub_lattice_interpolation(unittest.TestCase):

    def setUp(self):
        x, y = np.mgrid[0:100,0:100]
        position_list = np.vstack((x.flatten(), y.flatten())).T
        image_data = np.arange(100).reshape(10,10)
        self.sub_lattice = Sub_Lattice(position_list, image_data)

    def test_make_sub_lattice(self):
        sub_lattice = self.sub_lattice
        x_list = sub_lattice.x_position
        y_list = sub_lattice.y_position
        z_list = np.zeros_like(x_list)
        output = sub_lattice._get_regular_grid_from_unregular_property(
                x_list, y_list, z_list)
        z_interpolate = output[2]
        self.assertTrue(not z_interpolate.any())
