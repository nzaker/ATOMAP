import unittest
import numpy as np
import atomap.testing_tools as tt


class TestMakeTestData(unittest.TestCase):

    def test_simple_init(self):
        tt.MakeTestData(100, 100)

    def test_get_signal(self):
        imX0, imY0 = 100, 100
        test_data0 = tt.MakeTestData(imX0, imY0)
        self.assertEqual((imX0, imY0), test_data0.data_extent)
        self.assertEqual(test_data0.signal.axes_manager.shape, (imX0, imY0))
        self.assertFalse(test_data0.signal.data.any())

        imX1, imY1 = 100, 39
        test_data1 = tt.MakeTestData(imX1, imY1)
        self.assertEqual((imX1, imY1), test_data1.data_extent)
        self.assertEqual(test_data1.signal.axes_manager.shape, (imX1, imY1))

        imX2, imY2 = 34, 65
        test_data2 = tt.MakeTestData(imX2, imY2)
        self.assertEqual((imX2, imY2), test_data2.data_extent)
        self.assertEqual(test_data2.signal.axes_manager.shape, (imX2, imY2))

    def test_add_image_noise(self):
        test_data0 = tt.MakeTestData(1000, 1000)
        mu0, sigma0 = 0, 0.005
        test_data0.add_image_noise(mu=mu0, sigma=sigma0, only_positive=False)
        s0 = test_data0.signal
        self.assertAlmostEqual(s0.data.mean(), mu0, places=2)
        self.assertAlmostEqual(s0.data.std(), sigma0, places=2)

        test_data1 = tt.MakeTestData(1000, 1000)
        mu1, sigma1 = 10, 0.5
        test_data1.add_image_noise(mu=mu1, sigma=sigma1, only_positive=False)
        s1 = test_data1.signal
        self.assertAlmostEqual(s1.data.mean(), mu1, places=2)
        self.assertAlmostEqual(s1.data.std(), sigma1, places=2)

        test_data2 = tt.MakeTestData(1000, 1000)
        mu2, sigma2 = 154.2, 1.98
        test_data2.add_image_noise(mu=mu2, sigma=sigma2, only_positive=False)
        s2 = test_data2.signal
        self.assertAlmostEqual(s2.data.mean(), mu2, places=1)
        self.assertAlmostEqual(s2.data.std(), sigma2, places=1)

    def test_add_image_noise_only_positive(self):
        test_data0 = tt.MakeTestData(1000, 1000)
        test_data0.add_image_noise(mu=0, sigma=0.005, only_positive=True)
        s0 = test_data0.signal
        self.assertTrue((s0.data > 0).all())

    def test_add_image_noise_random_seed(self):
        test_data0 = tt.MakeTestData(100, 100)
        test_data0.add_image_noise(random_seed=0)
        s0 = test_data0.signal
        test_data1 = tt.MakeTestData(100, 100)
        test_data1.add_image_noise(random_seed=0)
        s1 = test_data1.signal
        self.assertTrue((s0.data == s1.data).all())

    def test_add_atom(self):
        x, y, sx, sy, A, r = 10, 5, 5, 9, 10, 2
        td = tt.MakeTestData(50, 50)
        td.add_atom(x, y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        self.assertEqual(len(td.sublattice.atom_list), 1)
        atom = td.sublattice.atom_list[0]
        self.assertEqual(atom.pixel_x, x)
        self.assertEqual(atom.pixel_y, y)
        self.assertEqual(atom.sigma_x, sx)
        self.assertEqual(atom.sigma_y, sy)
        self.assertEqual(atom.amplitude_gaussian, A)
        self.assertEqual(atom.rotation, r)

    def test_add_atom_list_simple(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        sx, sy, A, r = 2.1, 1.3, 9.5, 1.4
        td = tt.MakeTestData(100, 100)
        td.add_atom_list(
                x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        atom_list = td.sublattice.atom_list
        self.assertEqual(len(atom_list), len(x))
        for tx, ty, atom in zip(x, y, atom_list):
            self.assertEqual(atom.pixel_x, tx)
            self.assertEqual(atom.pixel_y, ty)
            self.assertEqual(atom.sigma_x, sx)
            self.assertEqual(atom.sigma_y, sy)
            self.assertEqual(atom.amplitude_gaussian, A)
            self.assertEqual(atom.rotation, r)

    def test_add_atom_list_all_lists(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        sx = np.random.random_sample(size=len(x))
        sy = np.random.random_sample(size=len(x))
        A = np.random.random_sample(size=len(x))
        r = np.random.random_sample(size=len(x))
        td = tt.MakeTestData(100, 100)
        td.add_atom_list(
                x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        atom_list = td.sublattice.atom_list
        self.assertEqual(len(atom_list), len(x))

        iterator = zip(x, y, sx, sy, A, r, atom_list)
        for tx, ty, tsx, tsy, tA, tr, atom in iterator:
            self.assertEqual(atom.pixel_x, tx)
            self.assertEqual(atom.pixel_y, ty)
            self.assertEqual(atom.sigma_x, tsx)
            self.assertEqual(atom.sigma_y, tsy)
            self.assertEqual(atom.amplitude_gaussian, tA)
            self.assertEqual(atom.rotation, tr)

    def test_add_atom_list_wrong_input(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        td = tt.MakeTestData(100, 100)
        with self.assertRaises(ValueError):
            td.add_atom_list(x, y[10:])

        sx = np.arange(10)
        with self.assertRaises(ValueError):
            td.add_atom_list(x, y, sigma_x=sx)

        sy = np.arange(20)
        with self.assertRaises(ValueError):
            td.add_atom_list(x, y, sigma_y=sy)

        A = np.arange(30)
        with self.assertRaises(ValueError):
            td.add_atom_list(x, y, amplitude=A)

        r = np.arange(5)
        with self.assertRaises(ValueError):
            td.add_atom_list(x, y, rotation=r)

    def test_gaussian_list(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        sx = np.random.random_sample(size=len(x))
        sy = np.random.random_sample(size=len(x))
        A = np.random.random_sample(size=len(x))
        r = np.random.random_sample(size=len(x))
        td = tt.MakeTestData(100, 100)
        td.add_atom_list(
                x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        gaussian_list = td.gaussian_list
        self.assertEqual(len(gaussian_list), len(x))

        iterator = zip(x, y, sx, sy, A, r, gaussian_list)
        for tx, ty, tsx, tsy, tA, tr, gaussian in iterator:
            self.assertEqual(gaussian.centre_x.value, tx)
            self.assertEqual(gaussian.centre_y.value, ty)
            self.assertEqual(gaussian.sigma_x.value, tsx)
            self.assertEqual(gaussian.sigma_y.value, tsy)
            self.assertEqual(gaussian.A.value, tA)
            self.assertEqual(gaussian.rotation.value, tr)


class TestMakeVectorTestGaussian(unittest.TestCase):
    def test_running(self):
        x, y, std, n = 10, 5, 0.5, 5000
        point_list = tt.make_vector_test_gaussian(
                x, y, standard_deviation=std, n=n)
        point_list_meanX = point_list[:, 0].mean()
        point_list_meanY = point_list[:, 1].mean()
        point_list_stdX = point_list[:, 0].std()
        point_list_stdY = point_list[:, 1].std()

        self.assertAlmostEqual(point_list_meanX, x, places=1)
        self.assertAlmostEqual(point_list_meanY, y, places=1)
        self.assertAlmostEqual(point_list_stdX, std, places=1)
        self.assertAlmostEqual(point_list_stdY, std, places=1)
        self.assertEqual(n, point_list.shape[0])


class TestMakeNnTestDataset(unittest.TestCase):
    def test_running(self):
        xN, yN, n = 4, 4, 60
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, n=n)

        total_point = n*(((2*xN)+1)*((2*yN)+1)-1)
        self.assertEqual(point_list.shape[0], total_point)
