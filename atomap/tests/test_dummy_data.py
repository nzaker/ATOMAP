import unittest
import atomap.dummy_data as dd


class test_dummy_data(unittest.TestCase):

    def test_make_simple_cubic_testdata(self):
        dd._make_simple_cubic_testdata()
        dd._make_simple_cubic_testdata(image_noise=False)
        dd._make_simple_cubic_testdata(image_noise=True)

    def test_get_simple_cubic_signal(self):
        dd.get_simple_cubic_signal()
        dd.get_simple_cubic_signal(image_noise=False)
        dd.get_simple_cubic_signal(image_noise=True)

    def test_get_simple_cubic_sublattice(self):
        dd.get_simple_cubic_sublattice()
        dd.get_simple_cubic_sublattice(image_noise=False)
        dd.get_simple_cubic_sublattice(image_noise=True)

    def test_get_two_sublattice_signal(self):
        dd.get_two_sublattice_signal()

    def test_get_simple_heterostructure_signal(self):
        dd.get_simple_heterostructure_signal()
        dd.get_simple_heterostructure_signal(image_noise=False)
        dd.get_simple_heterostructure_signal(image_noise=True)

    def test_get_dumbbell_signal(self):
        dd.get_dumbbell_signal()

    def test_get_fantasite(self):
        dd.get_fantasite()

    def test_get_fantasite_sublattice(self):
        dd.get_fantasite_sublattice()

    def test_get_perovskite110_ABF_signal(self):
        dd.get_perovskite110_ABF_signal()
        dd.get_perovskite110_ABF_signal(image_noise=False)
        dd.get_perovskite110_ABF_signal(image_noise=True)
