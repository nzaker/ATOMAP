import os
import unittest
import matplotlib
matplotlib.use('Agg')
from atomap.main import (
        make_atom_lattice_from_image,
        _get_signal_name)
from atomap.process_parameters import PerovskiteOxide110
from hyperspy.io import load
from hyperspy.signals import Signal2D

my_path = os.path.dirname(__file__)


class test_make_atom_lattice_from_image(unittest.TestCase):
    def setUp(self):
        s_adf_filename = os.path.join(
                my_path, "datasets", "test_ADF_cropped.hdf5")
        self.s_adf = load(s_adf_filename)
        self.pixel_separation = 19
        self.process_parameter = PerovskiteOxide110()

    def test_adf_image(self):
        s_adf = self.s_adf
        pixel_separation = self.pixel_separation
        process_parameter = self.process_parameter
        make_atom_lattice_from_image(
                s_adf,
                process_parameter=process_parameter,
                pixel_separation=pixel_separation)


class test_get_filename(unittest.TestCase):
    def setUp(self):
        self.s = Signal2D([range(10), range(10)])

    def test_empty_metadata_and_tmp_parameters(self):
        s = self.s.deepcopy()
        filename = _get_signal_name(s)
        self.assertEqual(filename, 'signal')

    def test_empty_metadata(self):
        s = self.s.deepcopy()
        s.__dict__['tmp_parameters']['filename'] = 'test2'
        filename = _get_signal_name(s)
        self.assertEqual(filename, 'test2')

    def test_metadata(self):
        s = self.s.deepcopy()
        s.__dict__['tmp_parameters']['filename'] = 'test2'
        s.metadata.General.title = 'test1'
        filename = _get_signal_name(s)
        self.assertEqual(filename, 'test1')
