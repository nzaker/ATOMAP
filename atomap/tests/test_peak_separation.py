import matplotlib
matplotlib.use('Agg')
import os
import unittest
from atomap.atom_finding_refining import get_feature_separation 
from hyperspy.api import load

my_path = os.path.dirname(__file__)

class test_peak_separation(unittest.TestCase):
    
    def setUp(self):
        s_adf_filename = os.path.join(
                my_path, "datasets", "test_ADF_cropped.hdf5")
        self.s_adf = load(s_adf_filename)

    def test_adf(self):
        s_adf = self.s_adf
        signal = get_feature_separation(
                s_adf,
                separation_range=(5, 7),
                separation_step=1,
                pca=True,
                subtract_background=True,
                normalize_intensity=True)
