import unittest
import atomap.dummy_data as dd


class test_dummy_data(unittest.TestCase):

    def test_get_simple_cubic_signal(self):
        dd.get_simple_cubic_signal()

    def test_get_simple_cubic_sublattice(self):
        dd.get_simple_cubic_sublattice()

    def test_get_two_sublattice_signal(self):
        dd.get_two_sublattice_signal()

    def test_get_simple_heterostructure_signal(self):
        dd.get_simple_heterostructure_signal()

    def test_get_dumbbell_signal(self):
        dd.get_dumbbell_signal()
