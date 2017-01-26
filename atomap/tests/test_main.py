import matplotlib
matplotlib.use('Agg')
import os
import unittest
from atomap.main import make_atom_lattice_from_image
from atomap.process_parameters import PerovskiteOxide110
from hyperspy.io import load

my_path = os.path.dirname(__file__)

class test_make_atom_lattice_from_image(unittest.TestCase):
    def setUp(self):
        s_adf_filename = os.path.join(
                my_path, "datasets", "test_ADF_cropped.hdf5")
        self.s_adf = load(s_adf_filename)
        self.pixel_separation = 19
        self.model_parameters = PerovskiteOxide110()

    def test_adf_image(self):
        s_adf = self.s_adf
        pixel_separation = self.pixel_separation
        model_parameters = self.model_parameters
        make_atom_lattice_from_image(
                s_adf,
                model_parameters=model_parameters,
                pixel_separation=pixel_separation)
