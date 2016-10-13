import matplotlib.pyplot as plt
import h5py
import numpy as np
from atomap.atomap_atom_finding_refining import\
        construct_zone_axes_from_sub_lattice


class Atom_Lattice():

    def __init__(self):
        self.sub_lattice_list = []
        self.adf_image = None
        self.inverted_abf_image = None
        self.path_name = ""

    def get_sub_lattice(self, sub_lattice_id):
        """
        Get a sublattice object from either sublattice index
        or name.
        """
        if isinstance(sub_lattice_id, str):
            for sub_lattice in self.sub_lattice_list:
                if sub_lattice.name == sub_lattice_id:
                    return(sub_lattice)
        elif isinstance(sub_lattice_id, int):
            return(self.sub_lattice_list[sub_lattice_id])
        raise ValueError('Could not find sub_lattice ' + str(sub_lattice_id))

    def construct_zone_axes_for_sub_lattices(self, sub_lattice_list=None):
        if sub_lattice_list is None:
            sub_lattice_list = self.sub_lattice_list
        for sub_lattice in sub_lattice_list:
            construct_zone_axes_from_sub_lattice(sub_lattice)

    def plot_all_sub_lattices(
            self,
            image=None,
            markersize=2,
            figname="all_sub_lattice.jpg"):
        if image is None:
            image = self.adf_image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.adf_image)
        for sub_lattice in self.sub_lattice_list:
            color = sub_lattice.plot_color
            for atom in sub_lattice.atom_list:
                ax.plot(
                        atom.pixel_x, atom.pixel_y,
                        'o', markersize=markersize, color=color)
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(figname)

    def plot_monolayer_distance_map(
            self,
            sub_lattice_list=None,
            interface_plane=None,
            max_number_of_zone_vectors=5):
        plt.ioff()
        if sub_lattice_list is None:
            sub_lattice_list = self.sub_lattice_list
        for sub_lattice in sub_lattice_list:
            sub_lattice.plot_monolayer_distance_map(
                interface_plane=interface_plane)

    def plot_atom_distance_map(
            self,
            sub_lattice_list=None,
            interface_plane=None,
            max_number_of_zone_vectors=5):
        plt.ioff()
        if sub_lattice_list is None:
            sub_lattice_list = self.sub_lattice_list
        for sub_lattice in sub_lattice_list:
            sub_lattice.plot_atom_distance_map(
                interface_plane=interface_plane)

    def plot_atom_distance_difference_map(
            self,
            sub_lattice_list=None,
            interface_plane=None,
            max_number_of_zone_vectors=5):
        plt.ioff()
        if sub_lattice_list is None:
            sub_lattice_list = self.sub_lattice_list
        for sub_lattice in sub_lattice_list:
            sub_lattice.plot_atom_distance_difference_map(
                interface_plane=interface_plane)

    def save_atom_lattice(self, filename=None):
        if filename is None:
            path = self.path_name
            filename = path + "/" + "atom_lattice.hdf5"

        h5f = h5py.File(filename, 'w')
        for sub_lattice in self.sub_lattice_list:
            subgroup_name = sub_lattice.tag + "_sub_lattice"
            modified_image_data = sub_lattice.adf_image
            original_image_data = sub_lattice.original_adf_image

            # Atom position data
            atom_positions = np.array([
                sub_lattice.x_position,
                sub_lattice.y_position]).swapaxes(0, 1)

#            atom_positions = np.array(sub_lattice._get_atom_position_list())
            sigma_x = np.array(sub_lattice.sigma_x)
            sigma_y = np.array(sub_lattice.sigma_y)
            rotation = np.array(sub_lattice.rotation)

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

            h5f[subgroup_name].attrs['pixel_size'] = sub_lattice.pixel_size
            h5f[subgroup_name].attrs['tag'] = sub_lattice.tag
            h5f[subgroup_name].attrs['path_name'] = sub_lattice.path_name
            h5f[subgroup_name].attrs['save_path'] = sub_lattice.save_path
            h5f[subgroup_name].attrs['plot_color'] = sub_lattice.plot_color

            # HDF5 does not supporting saving a list of strings, so converting
            # them to bytes
            zone_axis_names = sub_lattice.zones_axis_average_distances_names
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
