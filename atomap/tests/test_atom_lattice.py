import pytest
import numpy as np
from numpy.testing import assert_array_equal
import atomap.api as am
import atomap.atom_lattice as al
from atomap.testing_tools import MakeTestData
import atomap.initial_position_finding as ipf
import atomap.testing_tools as tt
import atomap.dummy_data as dd


class TestCreateAtomLatticeObject:

    def setup_method(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100, 100)
        peaks = np.arange(20).reshape(atoms_N, 2)
        self.sublattice = am.Sublattice(
                peaks,
                image_data)

    def test_create_empty_atom_lattice_object(self):
        am.Atom_Lattice()

    def test_create_atom_lattice_object(self):
        atom_lattice = am.Atom_Lattice()
        atom_lattice.sublattice_list.append(self.sublattice)

    def test_get_sublattice_atom_list_on_image(self):
        atom_lattice = am.Atom_Lattice()
        atom_lattice.image0 = self.sublattice.image
        atom_lattice.sublattice_list.append(self.sublattice)
        atom_lattice.get_sublattice_atom_list_on_image()

    def test_atom_lattice_wrong_input(self):
        with pytest.raises(ValueError):
            am.Atom_Lattice(self.sublattice.image, [self.sublattice, ])

    def test_atom_lattice_all_parameters(self):
        name = 'test_atom_lattice'
        atom_lattice = am.Atom_Lattice(self.sublattice.image, name=name,
                                       sublattice_list=[self.sublattice, ])
        assert atom_lattice.name == name
        assert (atom_lattice.image == self.sublattice.image).all()
        assert atom_lattice.sublattice_list == [self.sublattice, ]


class TestXYPosition:

    def setup_method(self):
        pos0 = np.array([[5, 10], [10, 15]])
        pos1 = np.array([[20, 25], [30, 35]])
        sublattice0 = am.Sublattice(pos0, np.zeros((40, 40)))
        sublattice1 = am.Sublattice(pos1, np.zeros((40, 40)))
        self.atom_lattice = am.Atom_Lattice(
                np.zeros((40, 40)), sublattice_list=[sublattice0, sublattice1])
        self.x_pos = np.concatenate((pos0[:, 0], pos1[:, 0]))
        self.y_pos = np.concatenate((pos0[:, 1], pos1[:, 1]))

    def test_x_position(self):
        assert (self.atom_lattice.x_position == self.x_pos).all()

    def test_y_position(self):
        assert (self.atom_lattice.y_position == self.y_pos).all()


class TestAtomLatticeIntegrate:

    def test_simple(self):
        atom_lattice = dd.get_simple_atom_lattice_two_sublattices()
        results = atom_lattice.integrate_column_intensity()
        assert len(results[0]) == len(atom_lattice.x_position)
        assert atom_lattice.image.shape == results[1].data.shape
        assert atom_lattice.image.shape == results[2].shape


class TestDumbbellLatticeInit:

    def test_empty(self):
        dumbbell_lattice = al.Dumbbell_Lattice()
        hasattr(dumbbell_lattice, 'plot')

    def test_two_sublattices(self):
        sublattice = am.Sublattice(np.zeros((10, 2)), np.zeros((10, 10)))
        sublattice_list = [sublattice, sublattice]
        dumbbell_lattice = al.Dumbbell_Lattice(
                image=np.zeros((10, 10)), sublattice_list=sublattice_list)
        hasattr(dumbbell_lattice, 'plot')

    def test_wrong_number_of_sublattices(self):
        sublattice = am.Sublattice(np.zeros((10, 2)), np.zeros((10, 10)))
        sublattice_list = [sublattice, ]
        with pytest.raises(ValueError):
            al.Dumbbell_Lattice(image=np.zeros((10, 10)),
                                sublattice_list=sublattice_list)

        sublattice_list = [sublattice, sublattice, sublattice]
        with pytest.raises(ValueError):
            al.Dumbbell_Lattice(image=np.zeros((10, 10)),
                                sublattice_list=sublattice_list)

    def test_wrong_number_of_atoms(self):
        sublattice0 = am.Sublattice(np.zeros((10, 2)), np.zeros((10, 10)))
        sublattice1 = am.Sublattice(np.zeros((11, 2)), np.zeros((10, 10)))
        sublattice_list = [sublattice0, sublattice1]
        with pytest.raises(ValueError):
            al.Dumbbell_Lattice(image=np.zeros((10, 10)),
                                sublattice_list=sublattice_list)


class TestDumbbellLattice:

    def setup_method(self):
        test_data = MakeTestData(200, 200)
        x0, y0 = np.mgrid[10:200:20, 10:200:20]
        x1, y1 = np.mgrid[16:200:20, 10:200:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        self.signal = test_data.signal

    def test_refine_position_gaussian(self):
        signal = self.signal
        atom_positions = am.get_atom_positions(signal, 4)
        vector = ipf.find_dumbbell_vector(atom_positions)
        dumbbell_positions = am.get_atom_positions(signal, separation=13)
        atom_lattice = ipf.make_atom_lattice_dumbbell_structure(
                signal, dumbbell_positions, vector)
        atom_lattice.refine_position_gaussian()


class TestAtomLatticePlot:

    def setup_method(self):
        test_data = tt.MakeTestData(50, 50)
        test_data.add_atom_list(np.arange(5, 45, 5), np.arange(5, 45, 5))
        self.atom_lattice = test_data.atom_lattice

    def test_plot(self):
        self.atom_lattice.plot()
        self.atom_lattice.plot(markersize=10, cmap='viridis')
        self.atom_lattice.plot(image=np.ones_like(self.atom_lattice.image0))


class TestAtomLatticeSignalProperty:

    def test_simple(self):
        atom_lattice = am.Atom_Lattice(np.ones((100, 100)))
        signal = atom_lattice.signal
        assert_array_equal(atom_lattice.image, signal.data)

    def test_no_image(self):
        atom_lattice = am.Atom_Lattice()
        with pytest.raises(ValueError):
            atom_lattice.signal

    def test_scaling(self):
        sublattice = am.Sublattice(
                [[10, 10], ], np.ones((20, 20)), pixel_size=0.2)
        atom_lattice = am.Atom_Lattice(
                np.ones((100, 100)), sublattice_list=[sublattice])
        signal = atom_lattice.signal
        assert signal.axes_manager.signal_axes[0].scale == 0.2
        assert signal.axes_manager.signal_axes[1].scale == 0.2
