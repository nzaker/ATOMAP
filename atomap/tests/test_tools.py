import unittest
from atomap.tools import array2signal
import numpy as np


class TestArray2Signal(unittest.TestCase):

    def test_array2signal(self):
        array = np.arange(100).reshape(10, 10)
        scale = 0.2
        s = array2signal(array, scale=scale)
        self.assertEqual(s.axes_manager[0].scale, scale)
        self.assertEqual(s.axes_manager[1].scale, scale)
        self.assertEqual(s.axes_manager.shape, array.shape)
