import h5py
from atom_lattice_class import Atom_Lattice
from sub_lattice_class import Sub_Lattice
from atomap_atom_finding_refining import construct_zone_axes_from_sub_lattice


def load_atom_lattice_from_hdf5(filename, construct_zone_axes=True):
    h5f = h5py.File(filename, 'r')
    atom_lattice = Atom_Lattice()
    for group_name in h5f:
        if ('atom_lattice' in group_name) or ('sub_lattice' in group_name):
            sub_lattice_set = h5f[group_name]
            modified_image_data = sub_lattice_set['modified_image_data'][:]
            original_image_data = sub_lattice_set['original_image_data'][:]
            atom_position_array = sub_lattice_set['atom_positions'][:]

            sub_lattice = Sub_Lattice(
                atom_position_array,
                modified_image_data)
            sub_lattice.original_adf_image = original_image_data

            if 'sigma_x' in sub_lattice_set.keys():
                sigma_x_array = sub_lattice_set['sigma_x'][:]
                for atom, sigma_x in zip(
                        sub_lattice.atom_list,
                        sigma_x_array):
                    atom.sigma_x = sigma_x
            if 'sigma_y' in sub_lattice_set.keys():
                sigma_y_array = sub_lattice_set['sigma_y'][:]
                for atom, sigma_y in zip(
                        sub_lattice.atom_list,
                        sigma_y_array):
                    atom.sigma_y = sigma_y
            if 'rotation' in sub_lattice_set.keys():
                rotation_array = sub_lattice_set['rotation'][:]
                for atom, rotation in zip(
                        sub_lattice.atom_list,
                        rotation_array):
                    atom.rotation = rotation

            sub_lattice.pixel_size = sub_lattice_set.attrs['pixel_size']
            sub_lattice.tag = sub_lattice_set.attrs['tag']
            sub_lattice.path_name = sub_lattice_set.attrs['path_name']
            sub_lattice.save_path = sub_lattice_set.attrs['save_path']
            sub_lattice.plot_color = sub_lattice_set.attrs['plot_color']

            if type(sub_lattice.tag) == bytes:
                sub_lattice.tag = sub_lattice.tag.decode()
            if type(sub_lattice.path_name) == bytes:
                sub_lattice.path_name = sub_lattice.path_name.decode()
            if type(sub_lattice.save_path) == bytes:
                sub_lattice.save_path = sub_lattice.save_path.decode()
            if type(sub_lattice.plot_color) == bytes:
                sub_lattice.plot_color = sub_lattice.plot_color.decode()

            atom_lattice.sub_lattice_list.append(sub_lattice)

            if construct_zone_axes:
                construct_zone_axes_from_sub_lattice(sub_lattice)

            if 'zone_axis_names_byte' in sub_lattice_set.keys():
                zone_axis_names_byte = sub_lattice_set.attrs[
                        'zone_axis_names_byte']
                zone_axis_names = []
                for zone_axis_name_byte in zone_axis_names_byte:
                    zone_axis_names.append(zone_axis_name_byte.decode())
                sub_lattice.zones_axis_average_distances_names = zone_axis_names

        if group_name == 'image_data0':
            atom_lattice.adf_image = h5f[group_name][:]

    atom_lattice.path_name = h5f.attrs['path_name']
    if type(atom_lattice.path_name) == bytes:
        atom_lattice.path_name = atom_lattice.path_name.decode()
    h5f.close()
    return(atom_lattice)
