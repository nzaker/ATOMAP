import h5py
from atomap.atom_lattice import Atom_Lattice
from atomap.sublattice import Sublattice
from atomap.atom_finding_refining import construct_zone_axes_from_sublattice


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
            sublattice.original_adf_image = original_image_data

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
            sublattice.tag = sublattice_set.attrs['tag']
            sublattice.path_name = sublattice_set.attrs['path_name']
            sublattice.save_path = sublattice_set.attrs['save_path']
            sublattice.plot_color = sublattice_set.attrs['plot_color']

            if type(sublattice.tag) == bytes:
                sublattice.tag = sublattice.tag.decode()
            if type(sublattice.path_name) == bytes:
                sublattice.path_name = sublattice.path_name.decode()
            if type(sublattice.save_path) == bytes:
                sublattice.save_path = sublattice.save_path.decode()
            if type(sublattice.plot_color) == bytes:
                sublattice.plot_color = sublattice.plot_color.decode()

            atom_lattice.sublattice_list.append(sublattice)

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
            atom_lattice.adf_image = h5f[group_name][:]

    atom_lattice.path_name = h5f.attrs['path_name']
    if type(atom_lattice.path_name) == bytes:
        atom_lattice.path_name = atom_lattice.path_name.decode()
    h5f.close()
    return(atom_lattice)
