import unittest
from atomap_tools import get_peak2d_skimage
from atomap_atom_finding_refining import subtract_average_background, do_pca_on_signal
from sub_lattice_class import Atom_Lattice
from hyperspy.api import load

class test_sub_lattice_classe(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = "tests/datasets/test_ADF_cropped.hdf5"
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        self.s_adf_modified = do_pca_on_signal(s_adf_modified)
        self.pixel_separation = peak_separation/s_adf.axes_manager[0].scale

    def test_make_sub_lattice(self):
        peaks = get_peak2d_skimage(
                self.s_adf_modified, 
                self.pixel_separation)[0]
        atom_lattice = Atom_Lattice(peaks, self.s_adf_modified)
