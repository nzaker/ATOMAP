import sys; sys.dont_write_bytecode = True
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import math
import operator
import copy
from scipy import ndimage
from scipy import interpolate
from matplotlib.gridspec import GridSpec
import os
import glob
import math
import json
from skimage.feature import peak_local_max
from scipy.stats import linregress
import h5py

from atomap_plotting import *

# Move to materials structure class
def save_material_structure(self, filename=None):
    if filename == None:
        path = self.path_name
        filename = path + "/" + "material_structure.hdf5"

    h5f = h5py.File(filename, 'w')
    for atom_lattice in self.atom_lattice_list:
        subgroup_name = atom_lattice.tag + "_atom_lattice"
        modified_image_data = atom_lattice.adf_image
        original_image_data = atom_lattice.original_adf_image

        # Atom position data
        atom_positions = np.array(atom_lattice._get_atom_position_list())
        sigma_x = np.array(atom_lattice.sigma_x)
        sigma_y = np.array(atom_lattice.sigma_y)
        rotation = np.array(atom_lattice.rotation)

        h5f.create_dataset(
                subgroup_name + "/modified_image_data",
                data=modified_image_data,
                chunks=True,
                compression='gzip')
        h5f.create_dataset(
                subgroup_name + "/original_image_data",
                data=original_image_data,
                chunks=True,
                compression='gzip')

        h5f.create_dataset(
                subgroup_name + "/atom_positions",
                data=atom_positions,
                chunks=True,
                compression='gzip')
        h5f.create_dataset(
                subgroup_name + "/sigma_x",
                data=sigma_x,
                chunks=True,
                compression='gzip')
        h5f.create_dataset(
                subgroup_name + "/sigma_y",
                data=sigma_y,
                chunks=True,
                compression='gzip')
        h5f.create_dataset(
                subgroup_name + "/rotation",
                data=rotation,
                chunks=True,
                compression='gzip')
        
        h5f[subgroup_name].attrs['pixel_size'] = atom_lattice.pixel_size
        h5f[subgroup_name].attrs['tag'] = atom_lattice.tag
        h5f[subgroup_name].attrs['path_name'] = atom_lattice.path_name
        h5f[subgroup_name].attrs['save_path'] = atom_lattice.save_path
        h5f[subgroup_name].attrs['plot_color'] = atom_lattice.plot_color

        # HDF5 does not supporting saving a list of strings, so converting
        # them to bytes
        zone_axis_names = atom_lattice.zones_axis_average_distances_names
        zone_axis_names_byte = []
        for zone_axis_name in zone_axis_names:
            zone_axis_names_byte.append(zone_axis_name.encode())
        h5f[subgroup_name].attrs['zone_axis_names_byte'] = zone_axis_names_byte

    h5f.create_dataset(
        "image_data0",
        data=self.adf_image,
        chunks=True,
        compression='gzip')
    h5f.attrs['path_name'] = self.path_name

    h5f.close()

def load_material_structure_from_hdf5(filename, construct_zone_axes=True):
    h5f = h5py.File(filename, 'r')
    material_structure = Material_Structure()
    for group_name in h5f:
        if 'atom_lattice' in group_name:
            atom_lattice_set = h5f[group_name]
            modified_image_data = atom_lattice_set['modified_image_data'][:]
            original_image_data = atom_lattice_set['original_image_data'][:]
            atom_position_array = atom_lattice_set['atom_positions'][:]

            atom_lattice = Atom_Lattice(
                atom_position_array,
                modified_image_data)
            atom_lattice.original_adf_image = original_image_data

            if 'sigma_x' in atom_lattice_set.keys():
                sigma_x_array = atom_lattice_set['sigma_x'][:]
                for atom, sigma_x in zip(atom_lattice.atom_list, sigma_x_array):
                    atom.sigma_x = sigma_x
            if 'sigma_y' in atom_lattice_set.keys():
                sigma_y_array = atom_lattice_set['sigma_y'][:]
                for atom, sigma_y in zip(atom_lattice.atom_list, sigma_y_array):
                    atom.sigma_y = sigma_y
            if 'rotation' in atom_lattice_set.keys():
                rotation_array = atom_lattice_set['rotation'][:]
                for atom, rotation in zip(atom_lattice.atom_list, rotation_array):
                    atom.rotation = rotation

            atom_lattice.pixel_size = atom_lattice_set.attrs['pixel_size']
            atom_lattice.tag = atom_lattice_set.attrs['tag']
            atom_lattice.path_name = atom_lattice_set.attrs['path_name']
            atom_lattice.save_path = atom_lattice_set.attrs['save_path']
            atom_lattice.plot_color = atom_lattice_set.attrs['plot_color']

            if type(atom_lattice.tag) == bytes:
                atom_lattice.tag = atom_lattice.tag.decode()
            if type(atom_lattice.path_name) == bytes:
                atom_lattice.path_name = atom_lattice.path_name.decode()
            if type(atom_lattice.save_path) == bytes:
                atom_lattice.save_path = atom_lattice.save_path.decode()
            if type(atom_lattice.plot_color) == bytes:
                atom_lattice.plot_color = atom_lattice.plot_color.decode()

            material_structure.atom_lattice_list.append(atom_lattice)

            if construct_zone_axes:
                construct_zone_axes_from_atom_lattice(atom_lattice)

            if 'zone_axis_names_byte' in atom_lattice_set.keys():
                zone_axis_names_byte = atom_lattice_set.attrs['zone_axis_names_byte']
                zone_axis_names = []
                for zone_axis_name_byte in zone_axis_names_byte:
                    zone_axis_names.append(zone_axis_name_byte.decode())
                atom_lattice.zones_axis_average_distances_names = zone_axis_names

        if group_name == 'image_data0':
            material_structure.adf_image = h5f[group_name][:]

    material_structure.path_name = h5f.attrs['path_name']
    if type(material_structure.path_name) == bytes:
        material_structure.path_name = material_structure.path_name.decode()
    h5f.close()
    return(material_structure)
