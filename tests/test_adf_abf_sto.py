import unittest
from atomap_tools import get_peak2d_skimage
from atomap_atom_finding_refining import subtract_average_background, do_pca_on_signal, construct_zone_axes_from_atom_lattice
from sub_lattice_class import Atom_Lattice
from hyperspy.api import load
import numpy as np

class test_adf_abf_sto(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "tests/datasets/test_ADF_cropped.hdf5"
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        self.s_adf_modified = do_pca_on_signal(s_adf_modified)
        self.pixel_size = s_adf.axes_manager[0].scale
        self.pixel_separation = peak_separation/self.pixel_size

        s_abf_filename = "tests/datasets/test_ABF_cropped.hdf5"
        s_abf = load(s_abf_filename)
        s_abf.change_dtype('float64')
        s_abf_modified = subtract_average_background(s_abf)

        self.peaks = get_peak2d_skimage(
                self.s_adf_modified, 
                self.pixel_separation)[0]

    def test_find_b_cation_atoms(self):
        a_atom_lattice = Atom_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        a_atom_lattice.pixel_size = self.pixel_size
        construct_zone_axes_from_atom_lattice(a_atom_lattice)

        zone_vector_100 = a_atom_lattice.zones_axis_average_distances[1]
        b_atom_list = a_atom_lattice.find_missing_atoms_from_zone_vector(
                zone_vector_100, new_atom_tag='B')
        b_atom_lattice = Atom_Lattice(
                b_atom_list, np.rot90(
                    np.fliplr(self.s_adf_modified.data)))
        b_atom_lattice.plot_atom_list_on_stem_data(
                image=b_atom_lattice.adf_image)
        self.assertEqual(len(b_atom_lattice.atom_list), 221)

    def test_find_b_atom_rows(self):
        a_atom_lattice = Atom_Lattice(
                self.peaks, 
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        a_atom_lattice.pixel_size = self.pixel_size
        construct_zone_axes_from_atom_lattice(a_atom_lattice)

        zone_vector_100 = a_atom_lattice.zones_axis_average_distances[1]
        b_atom_list = a_atom_lattice.find_missing_atoms_from_zone_vector(
                zone_vector_100, new_atom_tag='B')
        b_atom_lattice = Atom_Lattice(
                b_atom_list, np.rot90(
                    np.fliplr(self.s_adf_modified.data)))
        b_atom_lattice.pixel_size = self.pixel_size
        construct_zone_axes_from_atom_lattice(b_atom_lattice)
