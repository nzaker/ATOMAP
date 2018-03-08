import pytest
import numpy as np
import atomap.tools as to
from hyperspy.signals import Signal2D
from atomap.tools import array2signal1d, array2signal2d, Fingerprinter
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import atomap.dummy_data as dd
import atomap.testing_tools as tt


class TestArray2Signal:

    def test_array2signal2d(self):
        array = np.arange(100).reshape(10, 10)
        scale = 0.2
        s = array2signal2d(array, scale=scale)
        assert s.axes_manager[0].scale == scale
        assert s.axes_manager[1].scale == scale
        assert s.axes_manager.shape == array.shape

    def test_array2signal1d(self):
        array = np.arange(100)
        scale = 0.2
        s = array2signal1d(array, scale=scale)
        assert s.axes_manager[0].scale == scale
        assert s.axes_manager.shape == array.shape


class TestFingerprinter:

    def test_running_simple(self):
        xN, yN, n = 3, 3, 60
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        clusters = (xN*2+1)*(yN*2+1)-1
        assert fp.cluster_centers_.shape[0] == clusters

    def test_running_advanced(self):
        xN, yN, xS, yS, n = 2, 2, 9, 12, 70
        tolerance = 0.5
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, xS=xS, yS=yS, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        bool_list = []
        for ix in list(range(-xN, xN + 1)):
            for iy in list(range(-yN, yN + 1)):
                if (ix == 0) and (iy == 0):
                    pass
                else:
                    for point in point_list:
                        x, y = ix*xS, iy*yS
                        xD, yD = abs(x-point[0]), abs(y-point[1])
                        if (xD < tolerance) and (yD < tolerance):
                            bool_list.append(True)
                            break
        clusters = (xN*2+1)*(yN*2+1)-1
        assert len(bool_list) == clusters


class TestRemoveAtomsFromImageUsing2dGaussian:

    def setup_method(self):
        test_data = tt.MakeTestData(520, 520)
        x, y = np.mgrid[10:510:8j, 10:510:8j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        sublattice = test_data.sublattice
        sublattice.find_nearest_neighbors()
        self.sublattice = sublattice

    def test_running(self):
        sublattice = self.sublattice
        remove_atoms_from_image_using_2d_gaussian(
                sublattice.image,
                sublattice,
                percent_to_nn=0.40)


@pytest.mark.parametrize(
        "x,y,sx,sy,ox,oy", [
            (20, 20, 1, 1, 0, 0), (50, 20, 1, 1, 0, 0),
            (20, 20, 0.5, 2, 0, 0), (20, 20, 0.5, 2, 10, 5),
            (50, 20, 0.5, 4, -10, 20), (15, 50, 0.7, 2.3, -10, -5)])
def test_get_signal_centre(x, y, sx, sy, ox, oy):
    s = Signal2D(np.zeros((y, x)))
    print(s)
    am = s.axes_manager
    am[0].scale, am[0].offset, am[1].scale, am[1].offset = sx, ox, sy, oy
    xC, yC = to._get_signal_centre(s)
    print(xC, yC)
    assert xC == ((0.5*(x-1)*sx) + ox)
    assert yC == ((0.5*(y-1)*sy) + oy)


class TestRotatePointsAroundSignalCentre:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    @pytest.mark.parametrize("rot", [10, 30, 60, 90, 180, 250])
    def test_simple_rotation(self, rot):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
                self.s, self.x, self.y, rot)
        assert len(self.x) == len(x_rot)
        assert len(self.y) == len(y_rot)

    @pytest.mark.parametrize("rot", [0, 360])
    def test_zero_rotation(self, rot):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
                self.s, self.x, self.y, rot)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)

    def test_180_double_rotation(self):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
                self.s, self.x, self.y, 180)
        x_rot, y_rot = to.rotate_points_around_signal_centre(
                self.s, x_rot, y_rot, 180)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)

    def test_90_double_rotation(self):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
                self.s, self.x, self.y, 90)
        x_rot, y_rot = to.rotate_points_around_signal_centre(
                self.s, x_rot, y_rot, -90)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)


class TestRotatePointsAndSignal:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_rotation(self):
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(
                self.s, self.x, self.y, 180)
        assert self.s.axes_manager.shape == s_rot.axes_manager.shape
        assert len(self.x) == len(x_rot)
        assert len(self.y) == len(y_rot)

    def test_double_180_rotation(self):
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(
                self.s, self.x, self.y, 180)
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(
                s_rot, x_rot, y_rot, 180)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)


class TestFliplrPointsAroundSignalCentre:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_flip(self):
        x_flip, y_flip = to.fliplr_points_around_signal_centre(
                self.s, self.x, self.y)
        assert len(self.x) == len(x_flip)
        assert len(self.y) == len(y_flip)

    def test_flip_double(self):
        x_flip, y_flip = to.fliplr_points_around_signal_centre(
                self.s, self.x, self.y)
        x_flip, y_flip = to.fliplr_points_around_signal_centre(
                self.s, x_flip, y_flip)
        np.testing.assert_allclose(self.x, x_flip)
        np.testing.assert_allclose(self.y, y_flip)


class TestFliplrPointsAndSignal:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_flip(self):
        s_flip, x_flip, y_flip = to.fliplr_points_and_signal(
                self.s, self.x, self.y)
        s_flip, x_flip, y_flip = to.fliplr_points_and_signal(
                s_flip, x_flip, y_flip)
        np.testing.assert_allclose(self.s.data, s_flip.data)
        np.testing.assert_allclose(self.x, x_flip)
        np.testing.assert_allclose(self.y, y_flip)
