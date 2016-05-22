import h5py
from atom_lattice_class import Material_Structure 
from sub_lattice_class import Atom_Lattice
from atomap_atom_finding_refining import construct_zone_axes_from_atom_lattice

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
