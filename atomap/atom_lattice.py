import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from atomap.atom_finding_refining import\
        construct_zone_axes_from_sublattice


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

    def construct_zone_axes_for_sublattices(self, sublattice_list=None):
        if sublattice_list is None:
            sublattice_list = self.sublattice_list
        for sublattice in sublattice_list:
            construct_zone_axes_from_sublattice(sublattice)

    def plot_all_sublattices(
            self,
            image=None,
            markersize=4,
            figname="all_sublattice.jpg"):
        if image is None:
            image = self.adf_image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.adf_image)
        for sublattice in self.sublattice_list:
            color = sublattice.plot_color
            for atom in sublattice.atom_list:
                ax.plot(
                        atom.pixel_x, atom.pixel_y,
                        'o', markersize=markersize, color=color)
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        figname = os.path.join(self.path_name, figname)
        fig.savefig(figname)

    def plot_monolayer_distance_map(
            self,
            sublattice_list=None,
            interface_plane=None,
            max_number_of_zone_vectors=5):
        plt.ioff()
        if sublattice_list is None:
            sublattice_list = self.sublattice_list
        for sublattice in sublattice_list:
            sublattice.plot_monolayer_distance_map(
                interface_plane=interface_plane)

    def plot_atom_distance_map(
            self,
            sublattice_list=None,
            interface_plane=None,
            max_number_of_zone_vectors=5):
        plt.ioff()
        if sublattice_list is None:
            sublattice_list = self.sublattice_list
        for sublattice in sublattice_list:
            sublattice.plot_atom_distance_map(
                interface_plane=interface_plane)

    def plot_atom_distance_difference_map(
            self,
            sublattice_list=None,
            interface_plane=None,
            max_number_of_zone_vectors=5):
        plt.ioff()
        if sublattice_list is None:
            sublattice_list = self.sublattice_list
        for sublattice in sublattice_list:
            sublattice.plot_atom_distance_difference_map(
                interface_plane=interface_plane)

    def save_atom_lattice(self, filename=None):
        if filename is None:
            path = self.path_name
            filename = path + "/" + "atom_lattice.hdf5"

        h5f = h5py.File(filename, 'w')
        for sublattice in self.sublattice_list:
            subgroup_name = sublattice.tag + "_sublattice"
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
            h5f[subgroup_name].attrs['tag'] = sublattice.tag
            h5f[subgroup_name].attrs['path_name'] = sublattice.path_name
            h5f[subgroup_name].attrs['save_path'] = sublattice.save_path
            h5f[subgroup_name].attrs['plot_color'] = sublattice.plot_color

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
