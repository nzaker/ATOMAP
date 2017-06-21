import matplotlib
matplotlib.use('Agg')
import os
import unittest
import numpy as np
from atomap.atom_lattice import Atom_Lattice, Dumbbell_Lattice
from atomap.sublattice import Sublattice
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        construct_zone_axes_from_sublattice
from atomap.io import load_atom_lattice_from_hdf5
from atomap.testing_tools import make_artifical_atomic_signal
from atomap.initial_position_finding import (
        find_dumbbell_vector, make_atom_lattice_dumbbell_structure)
from atomap.atom_finding_refining import get_atom_positions
from hyperspy.api import load

my_path = os.path.dirname(__file__)

class test_create_atom_lattice_object(unittest.TestCase):

    def setUp(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(atoms_N,2)
        self.sublattice = Sublattice(
                peaks, 
                image_data)

    def test_create_empty_atom_lattice_object(self):
        atom_lattice = Atom_Lattice()

    def test_create_empty_atom_lattice_object(self):
        atom_lattice = Atom_Lattice()
        atom_lattice.sublattice_list.append(self.sublattice)

    def test_get_sublattice_atom_list_on_image(self):
        atom_lattice = Atom_Lattice()
        atom_lattice.image0 = self.sublattice.image
        atom_lattice.sublattice_list.append(self.sublattice)
        atom_lattice.get_sublattice_atom_list_on_image()


class test_atom_lattice_object_tools(unittest.TestCase):

    def setUp(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(atoms_N,2)
        sublattice0 = Sublattice(
                atom_position_list=peaks,
                image=image_data)
        sublattice1 = Sublattice(
                atom_position_list=peaks,
                image=image_data)
        self.atom_lattice = Atom_Lattice()
        self.atom_lattice.sublattice_list.extend([sublattice0, sublattice1])
        self.atom_lattice.image0 = image_data

    def test_save_atom_lattice(self):
        save_path = "test_atomic_lattice_save.hdf5"
        self.atom_lattice.save(
                filename=save_path, overwrite=True)

    def test_load_atom_lattice(self):
        hdf5_filename = os.path.join(my_path, "datasets", "test_atom_lattice.hdf5")
        load_atom_lattice_from_hdf5(hdf5_filename, construct_zone_axes=False)

    @unittest.expectedFailure
    def test_save_atom_lattice_already_exist(self):
        save_path = "test_atomic_lattice_save.hdf5"
        self.atom_lattice.save(
                filename=save_path, overwrite=True)
        self.atom_lattice.save(
                filename=save_path)


class test_dumbbell_lattice(unittest.TestCase):

    def setUp(self):
        x_list, y_list = [], []
        for x in range(10, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        for x in range(16, 200, 20):
            for y in range(10, 200, 20):
                x_list.append(x)
                y_list.append(y)
        sigma_value = 1
        sigma = [sigma_value]*len(x_list)
        A = [50]*len(x_list)
        s, g_list = make_artifical_atomic_signal(
                x_list, y_list, sigma_x=sigma, sigma_y=sigma, A=A, image_pad=0)
        vector = find_dumbbell_vector(s, 4)
        position_list = get_atom_positions(s, separation=13)
        atom_lattice = make_atom_lattice_dumbbell_structure(
                s, position_list, vector)
        self.atom_lattice = atom_lattice

    def test_refine_position_gaussian(self):
        atom_lattice = self.atom_lattice
        atom_lattice.refine_position_gaussian()
