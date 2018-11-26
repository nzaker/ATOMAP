from pytest import approx
import numpy as np
from numpy import pi
import math
from atomap.atom_position import Atom_Position
import atomap.testing_tools as tt
import atomap.dummy_data as dd


class TestCreateAtomPositionObject:

    def test_position(self):
        atom_x, atom_y = 10, 20
        atom_position = Atom_Position(atom_x, atom_y)
        assert atom_position.pixel_x == 10
        assert atom_position.pixel_y == 20

    def test_sigma(self):
        sigma_x, sigma_y = 2, 3
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        assert atom_position.sigma_x == sigma_x
        assert atom_position.sigma_y == sigma_y

        sigma_x, sigma_y = -5, -6
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        assert atom_position.sigma_x == abs(sigma_x)
        assert atom_position.sigma_y == abs(sigma_y)

    def test_amplitude(self):
        amplitude = 30
        atom_position = Atom_Position(1, 2, amplitude=amplitude)
        assert atom_position.amplitude_gaussian == amplitude

    def test_rotation(self):
        rotation0 = 0.0
        atom_position = Atom_Position(1, 2, rotation=rotation0)
        assert atom_position.rotation == rotation0

        rotation1 = math.pi/2
        atom_position = Atom_Position(1, 2, rotation=rotation1)
        assert atom_position.rotation == rotation1

        rotation2 = math.pi
        atom_position = Atom_Position(1, 2, rotation=rotation2)
        assert atom_position.rotation == 0

        rotation3 = math.pi*3/2
        atom_position = Atom_Position(1, 2, rotation=rotation3)
        assert atom_position.rotation == math.pi/2

        rotation4 = math.pi*2
        atom_position = Atom_Position(1, 2, rotation=rotation4)
        assert atom_position.rotation == 0


class TestAtomPositionObjectTools:

    def setup_method(self):
        self.atom_position = Atom_Position(1, 1)

    def test_get_atom_angle(self):
        atom_position0 = Atom_Position(1, 2)
        atom_position1 = Atom_Position(3, 1)
        atom_position2 = Atom_Position(1, 0)
        atom_position3 = Atom_Position(5, 1)
        atom_position4 = Atom_Position(2, 2)

        angle90 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position1)
        angle180 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position2)
        angle0 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position3)
        angle45 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position4)

        assert approx(angle90) == pi/2
        assert approx(angle180) == pi
        assert approx(angle0) == 0
        assert approx(angle45) == pi/4

    def test_as_gaussian(self):
        x, y, sx, sy, A, r = 10., 5., 2., 3.5, 9.9, 1.5
        atom_position = Atom_Position(
                x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        g = atom_position.as_gaussian()
        assert g.centre_x.value == x
        assert g.centre_y.value == y
        assert g.sigma_x.value == sx
        assert g.sigma_y.value == sy
        assert g.A.value == A
        assert g.rotation.value == r


class TestAtomPositionRefine:

    def test_center_of_mass_mask_radius(self):
        x, y, sx, sy = 15, 20, 2, 2
        test_data = tt.MakeTestData(50, 50)
        test_data.add_atom(x, y, sx, sy)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        image_data = test_data.signal.data
        atom.refine_position_using_center_of_mass(
                image_data, mask_radius=5)
        assert atom.pixel_x == approx(x)
        assert atom.pixel_y == approx(y)

    def test_2d_gaussian_mask_radius(self):
        x, y, sx, sy = 15, 20, 2, 2
        test_data = tt.MakeTestData(50, 50)
        test_data.add_atom(x, y, sx, sy)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        image_data = test_data.signal.data
        atom.refine_position_using_2d_gaussian(image_data, mask_radius=10)
        assert atom.pixel_x == approx(x)
        assert atom.pixel_y == approx(y)
        assert atom.sigma_x == approx(sx, rel=1e-4)
        assert atom.sigma_y == approx(sy, rel=1e-4)


class TestAtomPositionRefinePositionFlag:

    def test_one_atom_center_of_mass(self):
        test_data = tt.MakeTestData(20, 20)
        x, y = 10, 15
        test_data.add_atom(x, y)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        atom.pixel_x += 1
        atom.refine_position = False
        sublattice.refine_atom_positions_using_center_of_mass(mask_radius=5)
        assert atom.pixel_x == x + 1
        assert atom.pixel_y == y
        atom.refine_position = True
        sublattice.refine_atom_positions_using_center_of_mass(mask_radius=5)
        assert approx(atom.pixel_x, abs=0.001) == x
        assert approx(atom.pixel_y, abs=0.001) == y

    def test_one_atom_2d_gaussian_refine(self):
        test_data = tt.MakeTestData(20, 20)
        x, y = 10, 15
        test_data.add_atom(x, y)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        atom.pixel_x += 1
        atom.refine_position = False
        sublattice.refine_atom_positions_using_2d_gaussian(mask_radius=5)
        assert atom.pixel_x == x + 1
        assert atom.pixel_y == y
        atom.refine_position = True
        sublattice.refine_atom_positions_using_2d_gaussian(mask_radius=5)
        assert approx(atom.pixel_x) == x
        assert approx(atom.pixel_y) == y

    def test_many_atoms(self):
        sublattice = dd.get_simple_cubic_sublattice(
                image_noise=True)
        atom = sublattice.atom_list[0]
        atom.refine_position = False
        x_pos_orig = np.array(sublattice.x_position)
        y_pos_orig = np.array(sublattice.y_position)
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_2d_gaussian()
        assert atom.pixel_x == x_pos_orig[0]
        assert atom.pixel_y == x_pos_orig[1]
        assert not (x_pos_orig[1:] == sublattice.x_position[1:]).any()
        assert not (y_pos_orig[1:] == sublattice.y_position[1:]).any()
        sublattice.refine_atom_positions_using_2d_gaussian()
        assert atom.pixel_x == x_pos_orig[0]
        assert atom.pixel_y == x_pos_orig[1]
        assert not (x_pos_orig[1:] == sublattice.x_position[1:]).any()
        assert not (y_pos_orig[1:] == sublattice.y_position[1:]).any()
