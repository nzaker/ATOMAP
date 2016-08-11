import unittest
from atomap_tools import get_peak2d_skimage
from atomap_atom_finding_refining import subtract_average_background, do_pca_on_signal, construct_zone_axes_from_atom_lattice
from sub_lattice_class import Atom_Lattice
from hyperspy.api import load
import numpy as np
from atomap_atom_finding_refining import refine_atom_lattice

from atom_lattice_class import Material_Structure

class test_make_simple_sub_lattice(unittest.TestCase):
    def setUp(self):
        self.atoms_N = 10
        self.image_data = np.arange(10000).reshape(100,100)
        self.peaks = np.arange(20).reshape(self.atoms_N,2)

    def test_make_sub_lattice(self):
        sub_lattice = Atom_Lattice(
                self.peaks, 
                self.image_data)
        self.assertEqual(len(sub_lattice.atom_list), self.atoms_N)

class test_sub_lattice_processing(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "tests/datasets/test_ADF_cropped.hdf5"
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

    def test_make_sub_lattice(self):
        atom_lattice = Atom_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))

    def test_make_construct_zone_axes(self):
        atom_lattice = Atom_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        atom_lattice.pixel_size = self.pixel_size
        construct_zone_axes_from_atom_lattice(atom_lattice)

        number_zone_vector_110 = len(
                atom_lattice.atom_rows_by_zone_vector[
                    atom_lattice.zones_axis_average_distances[0]])
        number_zone_vector_100 = len(
                atom_lattice.atom_rows_by_zone_vector[
                    atom_lattice.zones_axis_average_distances[1]])

        self.assertEqual(number_zone_vector_110, 14)
        self.assertEqual(number_zone_vector_100, 17)

    def test_center_of_mass_refine(self):
        atom_lattice = Atom_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        atom_lattice.pixel_size = self.pixel_size
        refine_atom_lattice(
                atom_lattice, 
                [
                    (atom_lattice.adf_image, 1, 'center_of_mass')],
                0.25)
        atom_lattice.plot_atom_list_on_stem_data(
                image=atom_lattice.adf_image,
                figname="com_test.jpg")


class test_sub_lattice_distance_plot(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "tests/datasets/test_ADF_cropped.hdf5"
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
        self.atom_lattice = Atom_Lattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.atom_lattice.pixel_size = pixel_size

    def test_process_and_plot_distance_map(self):
        atom_lattice = self.atom_lattice
        construct_zone_axes_from_atom_lattice(atom_lattice)
        atom_lattice.plot_distance_map_for_all_zone_vectors()

class test_sub_lattice_distance_difference_plot(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "tests/datasets/test_ADF_cropped.hdf5"
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
        self.atom_lattice = Atom_Lattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        self.atom_lattice.pixel_size = pixel_size

    def test_process_and_plot_distance_differencemap(self):
        atom_lattice = self.atom_lattice
        construct_zone_axes_from_atom_lattice(atom_lattice)
        atom_lattice.plot_distance_difference_map_for_all_zone_vectors()

class test_sub_lattice_plotting(unittest.TestCase):
    def setUp(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(atoms_N,2)
        self.sub_lattice = Atom_Lattice(
                peaks, 
                image_data) 

    def test_plot_sub_lattice_atom_list(self):
        self.sub_lattice.plot_atom_list_on_stem_data(
                image=self.sub_lattice.adf_image)

class test_sub_lattice_interpolation(unittest.TestCase):
    def setUp(self):
        x, y = np.mgrid[0:100,0:100]
        position_list = np.vstack((x.flatten(), y.flatten())).T
        image_data = np.arange(100).reshape(10,10)
        self.sub_lattice = Atom_Lattice(position_list, image_data)

    def test_make_sub_lattice(self):
        sub_lattice = self.sub_lattice
        x_list = sub_lattice.x_position
        y_list = sub_lattice.y_position
        z_list = np.zeros_like(x_list)
        output = sub_lattice._get_regular_grid_from_unregular_property(
                x_list, y_list, z_list)
        z_interpolate = output[2]
        self.assertTrue(not z_interpolate.any())
