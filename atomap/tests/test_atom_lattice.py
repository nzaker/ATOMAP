import matplotlib
matplotlib.use('Agg')
import os
import unittest
import numpy as np
from atomap.atom_lattice import Atom_Lattice
from atomap.sublattice import Sublattice
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        construct_zone_axes_from_sublattice,\
        get_peak2d_skimage
from atomap.io import load_atom_lattice_from_hdf5
from hyperspy.api import load

my_path = os.path.dirname(__file__)

class test_create_atom_lattice_object(unittest.TestCase):

    def setUp(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(atoms_N,2)
        self.sublattice = Sublattice(
                peaks, 
                image_data)

    def test_create_empty_atom_lattice_object(self):
        atom_lattice = Atom_Lattice()

    def test_create_empty_atom_lattice_object(self):
        atom_lattice = Atom_Lattice()
        atom_lattice.sublattice_list.append(self.sublattice)

    def test_plot_all_sublattices(self):
        atom_lattice = Atom_Lattice()
        atom_lattice.adf_image = self.sublattice.adf_image
        atom_lattice.sublattice_list.append(self.sublattice)
        atom_lattice.plot_all_sublattices()

class test_atom_lattice_object_tools(unittest.TestCase):

    def setUp(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(atoms_N,2)
        sublattice = Sublattice(
                peaks, 
                image_data)
        sublattice.original_adf_image = image_data
        self.atom_lattice = Atom_Lattice()
        self.atom_lattice.sublattice_list.append(sublattice)
        self.atom_lattice.adf_image = image_data

    def test_save_atom_lattice(self):
        save_path = "test_atomic_lattice_save.hdf5"
        self.atom_lattice.save_atom_lattice(
                filename=save_path)

    def test_load_atom_lattice(self):
        hdf5_filename = os.path.join(my_path, "datasets", "test_atom_lattice.hdf5")
        load_atom_lattice_from_hdf5(hdf5_filename, construct_zone_axes=False)


class test_atom_lattice_plotting(unittest.TestCase):
    
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
        sublattice = Sublattice(
                peaks, 
                np.rot90(np.fliplr(s_adf_modified.data)))
        sublattice.pixel_size = pixel_size
        construct_zone_axes_from_sublattice(sublattice)

        self.atom_lattice = Atom_Lattice()
        self.atom_lattice.sublattice_list.append(sublattice)

    def test_plot_sublattice_monolayer_distance_map(self):
        self.atom_lattice.plot_monolayer_distance_map()

    def test_plot_sublattice_atom_distance_map(self):
        self.atom_lattice.plot_atom_distance_map()

    def test_plot_sublattice_atom_distance_difference_map(self):
        self.atom_lattice.plot_atom_distance_difference_map()
