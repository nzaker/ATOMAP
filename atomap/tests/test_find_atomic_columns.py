import matplotlib
matplotlib.use('Agg')
import os
import unittest
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        get_peak2d_skimage
from hyperspy.api import load
import matplotlib.pyplot as plt

my_path = os.path.dirname(__file__)

class test_finding_columns_skimage(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype('float64')
        s_adf_modified = subtract_average_background(s_adf)
        self.s_adf_modified = do_pca_on_signal(s_adf_modified)
        self.pixel_separation = peak_separation/s_adf.axes_manager[0].scale

    def test_find_number_of_columns(self):
        peaks = get_peak2d_skimage(
                self.s_adf_modified, 
                self.pixel_separation)[0]
        self.assertEqual(len(peaks), 238)
