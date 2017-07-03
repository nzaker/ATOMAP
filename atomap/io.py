import h5py
import os
from atomap.atom_lattice import Atom_Lattice
from atomap.sublattice import Sublattice
from atomap.atom_finding_refining import construct_zone_axes_from_sublattice
import numpy as np


def load_atom_lattice_from_hdf5(filename, construct_zone_axes=True):
    """
    Load an Atomap HDF5-file, restoring a saved Atom_Lattice.

    Parameters
    ----------
    filename : string
        Filename of the HDF5-file.
    construct_zone_axes : bool
        If True, find relations between atomic positions by
        constructing atomic planes. Default True.

    Returns
    -------
    Atomap Atom_Lattice object
    """
    h5f = h5py.File(filename, 'r')
    atom_lattice = Atom_Lattice()
    for group_name in h5f:
        if ('atom_lattice' in group_name) or ('sublattice' in group_name):
            sublattice_set = h5f[group_name]
            modified_image_data = sublattice_set['modified_image_data'][:]
            original_image_data = sublattice_set['original_image_data'][:]
            atom_position_array = sublattice_set['atom_positions'][:]

            sublattice = Sublattice(
                atom_position_array,
                modified_image_data)
            sublattice.original_image = original_image_data

            if 'sigma_x' in sublattice_set.keys():
                sigma_x_array = sublattice_set['sigma_x'][:]
                for atom, sigma_x in zip(
                        sublattice.atom_list,
                        sigma_x_array):
                    atom.sigma_x = sigma_x
            if 'sigma_y' in sublattice_set.keys():
                sigma_y_array = sublattice_set['sigma_y'][:]
                for atom, sigma_y in zip(
                        sublattice.atom_list,
                        sigma_y_array):
                    atom.sigma_y = sigma_y
            if 'rotation' in sublattice_set.keys():
                rotation_array = sublattice_set['rotation'][:]
                for atom, rotation in zip(
                        sublattice.atom_list,
                        rotation_array):
                    atom.rotation = rotation

            sublattice.pixel_size = sublattice_set.attrs['pixel_size']
            sublattice._tag = sublattice_set.attrs['tag']
            sublattice._plot_color = sublattice_set.attrs['plot_color']

            if type(sublattice._tag) == bytes:
                sublattice._tag = sublattice._tag.decode()
            if type(sublattice._plot_color) == bytes:
                sublattice._plot_color = sublattice._plot_color.decode()

            atom_lattice.sublattice_list.append(sublattice)

            if 'pixel_separation' in sublattice_set.attrs.keys():
                sublattice._pixel_separation = sublattice_set.attrs[
                        'pixel_separation']
            else:
                sublattice._pixel_separation = 0.8/sublattice.pixel_size

            if construct_zone_axes:
                construct_zone_axes_from_sublattice(sublattice)

            if 'zone_axis_names_byte' in sublattice_set.keys():
                zone_axis_list_byte = sublattice_set.attrs[
                        'zone_axis_names_byte']
                zone_axis_list = []
                for zone_axis_name_byte in zone_axis_list_byte:
                    zone_axis_list.append(zone_axis_name_byte.decode())
                sublattice.zones_axis_average_distances_names = zone_axis_list

        if group_name == 'image_data0':
            atom_lattice.image0 = h5f[group_name][:]
        if group_name == 'image_data1':
            atom_lattice.image1 = h5f[group_name][:]

    if 'name' in h5f.attrs.keys():
        atom_lattice.name = h5f.attrs['name']
    elif 'path_name' in h5f.attrs.keys():
        atom_lattice.name = h5f.attrs['path_name']
    if 'pixel_separation' in h5f.attrs.keys():
        atom_lattice._pixel_separation = h5f.attrs['pixel_separation']
    else:
        atom_lattice._pixel_separation = 0.8/sublattice.pixel_size
    if type(atom_lattice.name) == bytes:
        atom_lattice.name = atom_lattice.name.decode()
    h5f.close()
    return(atom_lattice)


def save_atom_lattice_to_hdf5(atom_lattice, filename, overwrite=False):
    if os.path.isfile(filename) and not overwrite:
        raise FileExistsError(
                "The file %s already exist, either change the name or "
                "use overwrite=True")
    elif os.path.isfile(filename) and overwrite:
        os.remove(filename)

    h5f = h5py.File(filename, 'w')
    for index, sublattice in enumerate(atom_lattice.sublattice_list):
        subgroup_name = sublattice._tag + "_sublattice"
        if subgroup_name in h5f:
            subgroup_name = str(index) + subgroup_name
        modified_image_data = sublattice.image
        original_image_data = sublattice.original_image

        # Atom position data
        atom_positions = np.array([
            sublattice.x_position,
            sublattice.y_position]).swapaxes(0, 1)

#            atom_positions = np.array(sublattice._get_atom_position_list())
        sigma_x = np.array(sublattice.sigma_x)
        sigma_y = np.array(sublattice.sigma_y)
        rotation = np.array(sublattice.rotation)

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

        h5f[subgroup_name].attrs['pixel_size'] = sublattice.pixel_size
        h5f[subgroup_name].attrs[
                'pixel_separation'] = sublattice._pixel_separation
        h5f[subgroup_name].attrs['tag'] = sublattice._tag
        h5f[subgroup_name].attrs['plot_color'] = sublattice._plot_color

        # HDF5 does not supporting saving a list of strings, so converting
        # them to bytes
        zone_axis_names = sublattice.zones_axis_average_distances_names
        zone_axis_names_byte = []
        for zone_axis_name in zone_axis_names:
            zone_axis_names_byte.append(zone_axis_name.encode())
        h5f[subgroup_name].attrs[
                'zone_axis_names_byte'] = zone_axis_names_byte

    h5f.create_dataset(
        "image_data0",
        data=atom_lattice.image0,
        chunks=True,
        compression='gzip')
    if hasattr(atom_lattice, 'image1'):
        h5f.create_dataset(
            "image_data1",
            data=atom_lattice.image1,
            chunks=True,
            compression='gzip')
    h5f.attrs['name'] = atom_lattice.name
    h5f.attrs['pixel_separation'] = atom_lattice._pixel_separation

    h5f.close()
