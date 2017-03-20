import matplotlib
matplotlib.use('Agg')
import unittest
from atomap.tools import array2signal1d, array2signal2d
import numpy as np


class TestArray2Signal(unittest.TestCase):

    def test_array2signal2d(self):
        array = np.arange(100).reshape(10, 10)
        scale = 0.2
        s = array2signal2d(array, scale=scale)
        self.assertEqual(s.axes_manager[0].scale, scale)
        self.assertEqual(s.axes_manager[1].scale, scale)
        self.assertEqual(s.axes_manager.shape, array.shape)

    def test_array2signal1d(self):
        array = np.arange(100)
        scale = 0.2
        s = array2signal1d(array, scale=scale)
        self.assertEqual(s.axes_manager[0].scale, scale)
        self.assertEqual(s.axes_manager.shape, array.shape)
