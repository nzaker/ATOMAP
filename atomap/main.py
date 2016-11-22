import os
import copy
import glob
import matplotlib.pyplot as plt
import hyperspy.api as hs
import numpy as np
from atomap.atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        refine_sublattice,\
        construct_zone_axes_from_sublattice,\
        get_peak2d_skimage,\
        normalize_signal

from atomap.tools import\
        remove_atoms_from_image_using_2d_gaussian

from atomap.atom_lattice import Atom_Lattice
from atomap.sublattice import Sublattice


class SublatticeParameterBase:
    def __init__(self):
        self.color = 'red'
        self.name = "Base Sublattice"
        self.sublattice_order = None

    def __repr__(self):
        return '<%s, %s>' % (
            self.__class__.__name__,
            self.name
            )


class GenericSublattice(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.color = 'red'
        self.tag = 'S0'
        self.image_type = 0
        self.name = "Sublattice 0"
        self.sublattice_order = 0
        self.refinement_config = {
                'config': [
                    ['image_data', 1, 'center_of_mass'],
                    ['image_data', 1, 'gaussian'],
                    ],
                'neighbor_distance': 0.35}


class PerovskiteOxide110SublatticeACation(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "A-cation"
        self.tag = "A"
        self.color = 'blue'
        self.image_type = 0
        self.zone_axis_list = [
                {'number': 0, 'name': '110'},
                {'number': 1, 'name': '100'},
                {'number': 2, 'name': '11-2'},
                {'number': 3, 'name': '112'},
                {'number': 5, 'name': '111'},
                {'number': 6, 'name': '11-1'},
                ]
        self.sublattice_order = 0
        self.refinement_config = {
                'config': [
                    ['image_data', 2, 'gaussian'],
                    ],
                'neighbor_distance': 0.35}


class PerovskiteOxide110SublatticeBCation(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "B-cation"
        self.tag = "B"
        self.color = 'green'
        self.image_type = 0
        self.zone_axis_list = [
                {'number': 0, 'name': '110'},
                {'number': 1, 'name': '100'},
                {'number': 2, 'name': '11-2'},
                {'number': 3, 'name': '112'},
                {'number': 5, 'name': '111'},
                {'number': 6, 'name': '11-1'}, ]
        self.sublattice_order = 1
        self.sublattice_position_sublattice = "A-cation"
        self.sublattice_position_zoneaxis = "100"
        self.refinement_config = {
                'config': [
                    ['image_data', 1, 'center_of_mass'],
                    ['image_data', 1, 'gaussian'],
                    ],
                'neighbor_distance': 0.25}
        self.atom_subtract_config = [
                {
                    'sublattice': 'A-cation',
                    'neighbor_distance': 0.35,
                    },
                ]


class PerovskiteOxide110SublatticeOxygen(SublatticeParameterBase):
    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "Oxygen"
        self.tag = "O"
        self.color = 'red'
        self.image_type = 1
        self.zone_axis_list = [
                {'number': 0, 'name': '110'},
                {'number': 1, 'name': '100'},
                {'number': 2, 'name': '11-2'},
                {'number': 3, 'name': '112'},
                {'number': 5, 'name': '111'},
                {'number': 6, 'name': '11-1'}, ]
        self.sublattice_order = 2
        self.sublattice_position_sublattice = "B-cation"
        self.sublattice_position_zoneaxis = "110"
        self.refinement_config = {
                'config': [
                    ['image_data', 1, 'center_of_mass'],
                    ['image_data', 1, 'gaussian'],
                    ],
                'neighbor_distance': 0.25}
        self.atom_subtract_config = [
                {
                    'sublattice': 'A-cation',
                    'neighbor_distance': 0.35,
                    },
                {
                    'sublattice': 'B-cation',
                    'neighbor_distance': 0.30,
                    },
                ]


class ModelParametersBase:
    def __init__(self):
        self.peak_separation = None
        self.name = None
        
        self.sublattice_list = []

    def __repr__(self):
        return '<%s, %s>' % (
            self.__class__.__name__,
            self.name,
            )

    def get_sublattice_from_order(self, order_number):
        for sublattice in self.sublattice_list:
            if order_number == sublattice.sublattice_order:
                return(sublattice)
        return(False)

    @property
    def number_of_sublattices(self):
        return(len(self.sublattice_list))


class GenericStructure(ModelParametersBase):
    def __init__(self):
        ModelParametersBase.__init__(self)
        self.peak_separation = None
        self.name = 'A structure'

        self.sublattice_list = [
            GenericSublattice(),
        ]


class PerovskiteOxide110(ModelParametersBase):
    def __init__(self):
        ModelParametersBase.__init__(self)
        self.name = "Peroskite 110"
        self.peak_separation = 0.127

        self.sublattice_list = [
            PerovskiteOxide110SublatticeACation(),
            PerovskiteOxide110SublatticeBCation(),
            PerovskiteOxide110SublatticeOxygen(),
            ]


class SrTiO3_110(PerovskiteOxide110):
    def __init__(self):
        PerovskiteOxide110.__init__(self)
        self.sublattice_names = "Sr", "Ti", "O"
        Ti_sublattice_position = {
                "sublattice": "Sr",
                "zoneaxis": "100"}
        O_sublattice_position = {
                "sublattice": "Ti",
                "zoneaxis": "110"}
        self.sublattice_position = [
                Ti_sublattice_position,
                O_sublattice_position]


def run_image_filtering(signal, invert_signal=False):
    signal.change_dtype('float64')
    signal_modified = subtract_average_background(signal)
    signal_modified = do_pca_on_signal(signal_modified)
    signal_modified = normalize_signal(
            signal_modified, invert_signal=invert_signal)
    if invert_signal:
        signal.data = 1./signal.data
    return(signal_modified)


def make_atom_lattice_from_image(
        s_image0,
        model_parameters=None,
        pixel_separation=None,
        s_image1=None):

    image0_filename = s_image0.__dict__['tmp_parameters']['filename']
    path_name = image0_filename
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    s_image0 = s_image0.deepcopy()
    s_image0_modified = run_image_filtering(s_image0)

    if model_parameters is None:
        model_parameters = GenericStructure()

    image0_scale = s_image0.axes_manager[0].scale
    if pixel_separation is None:
        if model_parameters.peak_separation is None:
            raise ValueError("pixel_separation is not set. Either set it in the model_parameters.peak_separation or pixel_separation parameter")
        else:
            pixel_separation = model_parameters.peak_separation/image0_scale
    initial_atom_position_list = get_peak2d_skimage(
            s_image0_modified,
            separation=pixel_separation)[0]

    if s_image1 is not None:
        s_image1 = s_image1.deepcopy()
        s_image1.data = 1./s_image1.data
        image1_data = np.rot90(np.fliplr(s_image1.data))

    #################################

    image0_data = np.rot90(np.fliplr(s_image0.data))
    image0_data_modified = np.rot90(np.fliplr(s_image0_modified.data))

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = image0_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = image0_data

    for sublattice_index in range(model_parameters.number_of_sublattices):
        sublattice_para = model_parameters.get_sublattice_from_order(
                sublattice_index)
        
        if sublattice_para.image_type == 0:
            s_image = s_image0
            image_data = image0_data
            image_data_modified = image0_data_modified
        if sublattice_para.image_type == 1:
            if s_image1 is not None:
                s_image = s_image1
                image_data = image1_data
                image_data_modified = image1_data
            else:
                break

        if sublattice_para.sublattice_order == 0:
            sublattice = Sublattice(
                initial_atom_position_list,
                image_data_modified)
        else:
            temp_sublattice = atom_lattice.get_sublattice(
                    sublattice_para.sublattice_position_sublattice)
            temp_zone_vector_index = temp_sublattice.get_zone_vector_index(
                    sublattice_para.sublattice_position_zoneaxis)
            zone_vector = temp_sublattice.zones_axis_average_distances[
                    temp_zone_vector_index]
            atom_list = temp_sublattice.find_missing_atoms_from_zone_vector(
                    zone_vector, new_atom_tag=sublattice_para.tag)

            sublattice = Sublattice(
                atom_list,
                image_data)

        sublattice.save_path = "./" + path_name + "/"
        sublattice.path_name = path_name
        sublattice.plot_color = sublattice_para.color
        sublattice.name = sublattice_para.name
        sublattice.tag = sublattice_para.tag
        sublattice.pixel_size = s_image.axes_manager[0].scale
        sublattice.original_adf_image = image_data
        atom_lattice.sublattice_list.append(sublattice)

        for atom in sublattice.atom_list:
            atom.sigma_x = 0.05/sublattice.pixel_size
            atom.sigma_y = 0.05/sublattice.pixel_size

        if not(sublattice_para.sublattice_order == 0):
            construct_zone_axes_from_sublattice(sublattice)
            atom_subtract_config = sublattice_para.atom_subtract_config
            image_data = sublattice.adf_image
            for atom_subtract_para in atom_subtract_config:
                temp_sublattice = atom_lattice.get_sublattice(
                        atom_subtract_para['sublattice'])
                neighbor_distance = atom_subtract_para['neighbor_distance']
                image_data = remove_atoms_from_image_using_2d_gaussian(
                    image_data,
                    temp_sublattice,
                    percent_to_nn=neighbor_distance)
            sublattice.adf_image = image_data
            sublattice.original_adf_image = image_data

        refinement_config = sublattice_para.refinement_config
        refinement_neighbor_distance = refinement_config['neighbor_distance']
        refinement_steps = refinement_config['config']
        for refinement_step in refinement_steps:
            if refinement_step[0] == 'image_data':
                refinement_step[0] = sublattice.original_adf_image
            elif refinement_step[0] == 'image_data_modified':
                refinement_step[0] = sublattice.adf_image
            else:
                refinement_step[0] = sublattice.original_adf_image

        refine_sublattice(
            sublattice,
            refinement_steps,
            refinement_neighbor_distance)
        construct_zone_axes_from_sublattice(sublattice)

        if hasattr(sublattice_para, 'zone_axis_list'):
            for zone_axis in sublattice_para.zone_axis_list:
                if zone_axis['number'] <= len(
                        sublattice.zones_axis_average_distances_names):
                    sublattice.zones_axis_average_distances_names[
                            zone_axis['number']] =\
                            zone_axis['name']

    return(atom_lattice)
