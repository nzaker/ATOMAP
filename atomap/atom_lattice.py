import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from atomap.atom_finding_refining import\
        construct_zone_axes_from_sublattice
from atomap.plotting import _make_atom_position_marker_list
from atomap.tools import array2signal2d

class Atom_Lattice():

    def __init__(self):
        self.sublattice_list = []
        self.adf_image = None
        self.inverted_abf_image = None
        self.path_name = ""

    def __repr__(self):
        return '<%s, %s (sublattice(s): %s)>' % (
            self.__class__.__name__,
            self.path_name,
            len(self.sublattice_list),
            )

    def get_sublattice(self, sublattice_id):
        """
        Get a sublattice object from either sublattice index
        or name.
        """
        if isinstance(sublattice_id, str):
            for sublattice in self.sublattice_list:
                if sublattice.name == sublattice_id:
                    return(sublattice)
        elif isinstance(sublattice_id, int):
            return(self.sublattice_list[sublattice_id])
        raise ValueError('Could not find sublattice ' + str(sublattice_id))

    def _construct_zone_axes_for_sublattices(self, sublattice_list=None):
        if sublattice_list is None:
            sublattice_list = self.sublattice_list
        for sublattice in sublattice_list:
            construct_zone_axes_from_sublattice(sublattice)

    def get_sublattice_atom_list_on_image(
            self,
            image=None,
            color='red',
            add_numbers=False,
            markersize=20):
        if image is None:
            image = self.adf_image
        marker_list = []
        scale = self.sublattice_list[0].pixel_size
        for sublattice in self.sublattice_list:
            marker_list.extend(_make_atom_position_marker_list(
                    sublattice.atom_list,
                    scale=scale,
                    color=sublattice._plot_color,
                    markersize=markersize,
                    add_numbers=add_numbers))
        signal = array2signal2d(image, scale)
        signal.add_marker(marker_list, permanent=True, plot_marker=False)

        return signal

    def save_atom_lattice(self, filename=None):
        if filename is None:
            path = self.path_name
            filename = path + "/" + "atom_lattice.hdf5"

        h5f = h5py.File(filename, 'w')
        for sublattice in self.sublattice_list:
            subgroup_name = sublattice._tag + "_sublattice"
            modified_image_data = sublattice.adf_image
            original_image_data = sublattice.original_adf_image

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
            h5f[subgroup_name].attrs['tag'] = sublattice._tag
            h5f[subgroup_name].attrs['path_name'] = sublattice.path_name
            h5f[subgroup_name].attrs['save_path'] = sublattice._save_path
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
            data=self.adf_image,
            chunks=True,
            compression='gzip')
        h5f.attrs['path_name'] = self.path_name

        h5f.close()
