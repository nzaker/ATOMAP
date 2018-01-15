import unittest
import atomap.dummy_data as dd


class test_dummy_data(unittest.TestCase):

    def test_make_simple_cubic_testdata(self):
        dd._make_simple_cubic_testdata()
        dd._make_simple_cubic_testdata(image_noise=False)
        dd._make_simple_cubic_testdata(image_noise=True)

    def test_get_simple_cubic_signal(self):
        s0 = dd.get_simple_cubic_signal()
        s0.plot()
        s1 = dd.get_simple_cubic_signal(image_noise=False)
        s1.plot()
        s2 = dd.get_simple_cubic_signal(image_noise=True)
        s2.plot()

    def test_get_simple_cubic_sublattice(self):
        s0 = dd.get_simple_cubic_sublattice()
        s0.plot()
        s1 = dd.get_simple_cubic_sublattice(image_noise=False)
        s1.plot()
        s2 = dd.get_simple_cubic_sublattice(image_noise=True)
        s2.plot()

    def test_get_two_sublattice_signal(self):
        s = dd.get_two_sublattice_signal()
        s.plot()

    def test_get_simple_heterostructure_signal(self):
        s0 = dd.get_simple_heterostructure_signal()
        s0.plot()
        s1 = dd.get_simple_heterostructure_signal(image_noise=False)
        s1.plot()
        s2 = dd.get_simple_heterostructure_signal(image_noise=True)
        s2.plot()

    def test_get_dumbbell_signal(self):
        s = dd.get_dumbbell_signal()
        s.plot()

    def test_get_perovskite110_ABF_signal(self):
        s0 = dd.get_perovskite110_ABF_signal()
        s0.plot()
        s1 = dd.get_perovskite110_ABF_signal(image_noise=False)
        s1.plot()
        s2 = dd.get_perovskite110_ABF_signal(image_noise=True)
        s2.plot()

    def test_get_simple_atom_lattice_two_sublattices(self):
        s0 = dd.get_simple_atom_lattice_two_sublattices()
        s0.plot()
        s1 = dd.get_simple_atom_lattice_two_sublattices(image_noise=True)
        s1.plot()
        s2 = dd.get_simple_atom_lattice_two_sublattices(image_noise=False)
        s2.plot()


class dummy_data_fantasite(unittest.TestCase):

    def test_signal(self):
        s = dd.get_fantasite()
        s.plot()
        s1 = dd.get_fantasite()
        self.assertTrue((s.data == s1.data).all())

    def test_sublattice(self):
        sublattice = dd.get_fantasite_sublattice()
        self.assertEqual(
                len(sublattice.x_position), len(sublattice.y_position))

    def test_atom_lattice(self):
        atom_lattice = dd.get_fantasite_atom_lattice()
        self.assertEqual(len(atom_lattice.sublattice_list), 2)
