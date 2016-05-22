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

# Rename to Atom_Lattice
class Material_Structure():
    def __init__(self):
        self.atom_lattice_list = []
        self.adf_image = None
        self.inverted_abf_image = None

    def construct_zone_axes_for_atom_lattices(self, atom_lattice_list=None):
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            construct_zone_axes_from_atom_lattice(atom_lattice)

    def plot_all_atom_lattices(self, image=None, markersize=1, figname="all_atom_lattice.jpg"):
        if image == None:
            image = self.adf_image
        fig, ax = plt.subplots(figsize=(10,10))
        cax = ax.imshow(self.adf_image)
        for atom_lattice in self.atom_lattice_list:
            color = atom_lattice.plot_color
            for atom in atom_lattice.atom_list:
                ax.plot(atom.pixel_x, atom.pixel_y, 'o', markersize=markersize, color=color)
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(figname)

# To be removed
#    def plot_all_atom_lattices_abf(self, markersize=1, figname="all_atom_lattice_abf.jpg"):
#        plt.ioff()
#        fig, ax = plt.subplots(figsize=(10,10))
#        cax = ax.imshow(self.abf_image)
#        for atom_lattice in self.atom_lattice_list:
#            color = atom_lattice.plot_color
#            for atom in atom_lattice.atom_list:
#                ax.plot(atom.pixel_x, atom.pixel_y, 'o', markersize=markersize, color=color)
#        ax.set_ylim(0, self.abf_image.shape[0])
#        ax.set_xlim(0, self.abf_image.shape[1])
#        fig.savefig(figname)

    def plot_atom_distance_maps_for_zone_vectors_and_lattices(
            self,
            atom_lattice_list=None,
            interface_row=None,
            max_number_of_zone_vectors=1):
        plt.ioff()
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            atom_lattice.plot_distance_map_for_all_zone_vectors(
                atom_row_marker=interface_row,
                atom_list=atom_lattice.atom_list,
                max_number_of_zone_vectors=max_number_of_zone_vectors)

    def plot_atom_distance_difference_maps_for_zone_vectors_and_lattices(
            self,
            atom_list_as_zero=None,
            atom_lattice_list=None):
        plt.ioff()
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            atom_lattice.plot_distance_difference_map_for_all_zone_vectors(
                    atom_list_as_zero=atom_list_as_zero)
 
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
