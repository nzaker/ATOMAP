import unittest
import numpy as np
import math
import atomap.quantification as quant
from atomap.example_data import get_detector_image_signal
from atomap.dummy_data import get_simple_cubic_signal


class TestDetectorNormalisation(unittest.TestCase):

    def test_centered_distance_matrix(self):
        s = quant.centeredDistanceMatrix((32, 32), np.zeros((64, 64)))
        self.assertEqual(s[32, 32], 1)
        self.assertEqual(s[63, 63], np.sqrt((63-31)**2 + (63-32)**2))

    def test_detector_threshold(self):
        det_image = get_detector_image_signal()
        threshold_image = quant._detector_threshold(det_image.data)
        self.assertNotEqual(np.sum(threshold_image), 0)
        self.assertEqual(det_image.data.shape, threshold_image.shape)

    def test_radial_profile(self):
        det_image = get_detector_image_signal()
        profile = quant._radial_profile(det_image.data, (256, 256))
        self.assertEqual(len(np.shape(profile)), 1)
        self.assertEqual(np.shape(profile)[0], math.ceil(math.sqrt(2) * 256))

    def test_detector_normalisation(self):
        det_image = get_detector_image_signal()
        img = get_simple_cubic_signal(image_noise=True)
        img = (img) * 300000 + 4000
        image_normalised = quant.detector_normalisation(img, det_image, 60)
        self.assertTrue(image_normalised.data.max() < 1)
        self.assertEqual(image_normalised.data.shape, img.data.shape)

    # def test_analyse_flux(self):
        # need example flux profile for this test
