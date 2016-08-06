import unittest
import numpy as np
from atom_lattice_class import Material_Structure
from sub_lattice_class import Atom_Lattice
from atomap_io import load_material_structure_from_hdf5

class test_create_atom_lattice_object(unittest.TestCase):

    def setUp(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(atoms_N,2)
        self.sub_lattice = Atom_Lattice(
                peaks, 
                image_data)

    def test_create_empty_atom_lattice_object(self):
        atom_lattice = Material_Structure()

    def test_create_empty_atom_lattice_object(self):
        atom_lattice = Material_Structure()
        atom_lattice.atom_lattice_list.append(self.sub_lattice)

    def test_plot_all_sub_lattices(self):
        atom_lattice = Material_Structure()
        atom_lattice.adf_image = self.sub_lattice.adf_image
        atom_lattice.atom_lattice_list.append(self.sub_lattice)
        atom_lattice.plot_all_atom_lattices()

class test_atom_lattice_object_tools(unittest.TestCase):

    def setUp(self):
        atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(atoms_N,2)
        sub_lattice = Atom_Lattice(
                peaks, 
                image_data)
        sub_lattice.original_adf_image = image_data
        self.atom_lattice = Material_Structure()
        self.atom_lattice.atom_lattice_list.append(sub_lattice)
        self.atom_lattice.adf_image = image_data

    def test_save_atom_lattice(self):
        save_path = "test_atomic_lattice_save.hdf5"
        self.atom_lattice.save_material_structure(
                filename=save_path)

    def test_load_atom_lattice(self):
        hdf5_filename = "tests/datasets/test_atom_lattice.hdf5"
        load_material_structure_from_hdf5(
                hdf5_filename, 
                construct_zone_axes=False)
