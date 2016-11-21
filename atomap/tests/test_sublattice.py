import matplotlib
matplotlib.use('Agg')
import os
import unittest
import numpy as np
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        construct_zone_axes_from_sublattice,\
        get_peak2d_skimage
from atomap.sublattice import Sublattice
from hyperspy.api import load
from atomap.atom_finding_refining import refine_sublattice

from atomap.atom_lattice import Atom_Lattice

my_path = os.path.dirname(__file__)

class test_make_simple_sublattice(unittest.TestCase):
    def setUp(self):
        self.atoms_N = 10
        self.image_data = np.arange(10000).reshape(100,100)
        self.peaks = np.arange(20).reshape(self.atoms_N,2)

    def test_make_sublattice(self):
        sublattice = Sublattice(
                self.peaks, 
                self.image_data)
        self.assertEqual(len(sublattice.atom_list), self.atoms_N)


class test_sublattice_construct_refine(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        self.s_adf_modified = do_pca_on_signal(s_adf_modified)
        self.pixel_size = s_adf.axes_manager[0].scale
        self.pixel_separation = peak_separation/self.pixel_size

        self.peaks = get_peak2d_skimage(
                self.s_adf_modified, 
                self.pixel_separation)[0]

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

    def test_center_of_mass_refine(self):
        sublattice = Sublattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        sublattice.pixel_size = self.pixel_size
        refine_sublattice(
                sublattice, 
                [
                    (sublattice.adf_image, 1, 'center_of_mass')],
                0.25)
        sublattice.plot_atom_list_on_image_data(
                image=sublattice.adf_image,
                figname="com_test.jpg")


class test_sublattice_processing(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sublattice = Sublattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(self.sublattice)

    def test_zone_vector_mean_angle(self):
        zone_vector = self.sublattice.zones_axis_average_distances[0]
        mean_angle = self.sublattice.get_zone_vector_mean_angle(zone_vector)

    def test_get_nearest_neighbor_directions(self):
        self.sublattice.get_nearest_neighbor_directions()

class test_sublattice_plotting_distance(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sublattice = Sublattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(self.sublattice)

    def test_plot_sublattice_atom_distance_map(self):
        self.sublattice.plot_atom_distance_map()

class test_sublattice_plotting_distance_difference(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sublattice = Sublattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(self.sublattice)

    def test_plot_sublattice_atom_distance_difference_map(self):
        self.sublattice.plot_atom_distance_difference_map()

class test_sublattice_plotting_monolayer_distance(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sublattice = Sublattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(self.sublattice)

    def test_plot_sublattice_monolayer_distance_map(self):
        self.sublattice.plot_monolayer_distance_map()

class test_sublattice_plotting_atom_list(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sublattice = Sublattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(self.sublattice)

    def test_plot_sublattice_atom_list(self):
        self.sublattice.plot_atom_list_on_image_data(
                image=self.sublattice.adf_image)

class test_sublattice_plotting_ellipticity(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
        pixel_size = s_adf.axes_manager[0].scale
        pixel_separation = peak_separation/pixel_size

        peaks = get_peak2d_skimage(
                s_adf_modified, 
                pixel_separation)[0]
        self.sublattice = Sublattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(self.sublattice)

    def test_plot_ellipticity_rotation_complex(self):
        self.sublattice.plot_ellipticity_rotation_complex()

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
