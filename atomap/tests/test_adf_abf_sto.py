import matplotlib
matplotlib.use('Agg')
import os
import unittest
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        construct_zone_axes_from_sublattice,\
        get_atom_positions

from atomap.sublattice import Sublattice
from atomap.atom_lattice import Atom_Lattice
from hyperspy.api import load
import numpy as np

my_path = os.path.dirname(__file__)


class test_adf_abf_sto_autoprocess(unittest.TestCase):

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

        s_abf_filename = os.path.join(
                my_path, "datasets", "test_ABF_cropped.hdf5")
        s_abf = load(s_abf_filename)
        s_abf.change_dtype('float64')
        s_abf_modified = subtract_average_background(s_abf)

        self.peaks = get_atom_positions(
                self.s_adf_modified,
                self.pixel_separation)

    def test_find_b_cation_atoms(self):
        a_sublattice = Sublattice(
                self.peaks,
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        a_sublattice.pixel_size = self.pixel_size
        construct_zone_axes_from_sublattice(a_sublattice)

        zone_vector_100 = a_sublattice.zones_axis_average_distances[1]
        b_atom_list = a_sublattice._find_missing_atoms_from_zone_vector(
                zone_vector_100, new_atom_tag='B')
        b_sublattice = Sublattice(
                b_atom_list, np.rot90(
                    np.fliplr(self.s_adf_modified.data)))
        self.assertEqual(len(b_sublattice.atom_list), 221)

    def test_find_b_atom_planes(self):
        a_sublattice = Sublattice(
                self.peaks,
                np.rot90(np.fliplr(self.s_adf_modified.data)))
        a_sublattice.pixel_size = self.pixel_size
        construct_zone_axes_from_sublattice(a_sublattice)

        zone_vector_100 = a_sublattice.zones_axis_average_distances[1]
        b_atom_list = a_sublattice._find_missing_atoms_from_zone_vector(
                zone_vector_100, new_atom_tag='B')
        b_sublattice = Sublattice(
                b_atom_list, np.rot90(
                    np.fliplr(self.s_adf_modified.data)))
        b_sublattice.pixel_size = self.pixel_size
        construct_zone_axes_from_sublattice(b_sublattice)


class test_adf_abf_sto_manualprocess(unittest.TestCase):

    def test_manual_processing(self):
        s_adf_filename = os.path.join(
                my_path, "datasets", "test_ADF_cropped.hdf5")
        s = load(s_adf_filename)
        atom_positions = get_atom_positions(
                signal=s,
                separation=17,
                threshold_rel=0.02,
                )
        sublattice = Sublattice(
                atom_position_list=atom_positions,
                image=s.data)
        sublattice._find_nearest_neighbors()
        sublattice.refine_atom_positions_using_center_of_mass(
                sublattice.image, percent_to_nn=0.4)
        sublattice.refine_atom_positions_using_2d_gaussian(
                sublattice.image, percent_to_nn=0.4)

        atom_lattice = Atom_Lattice(image=s.data, sublattice_list=[sublattice])
        sublattice.construct_zone_axes()
