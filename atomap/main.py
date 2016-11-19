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
        image0_filename,
        model_parameters=None,
        pixel_separation=None,
        image1_filename=None):
    s_image0 = hs.load(image0_filename)
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

    if image1_filename is not None:
        s_image1 = hs.load(image1_filename)
        s_image1.data = 1./s_image1.data
        image1_data = np.rot90(np.fliplr(s_image1.data))

    #################################
    path_name = image0_filename
    path_name = path_name[0: path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)

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
            if image1_filename is not None:
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


def run_atom_lattice_peakfinding_process(
        s_adf_filename,
        s_abf_filename,
        model_parameters="Perovskite110",
        refinement_interation_config=None,
        invert_abf_signal=True):
    plt.ioff()
    ################
    s_abf = hs.load(s_abf_filename)
    s_abf_modified = run_image_filtering(s_abf, invert_signal=True)

    s_adf = hs.load(s_adf_filename)
    s_adf_modified = run_image_filtering(s_adf)

    if model_parameters == "Perovskite110":
        model_parameters = PerovskiteOxide110()

    s_adf_scale = s_adf.axes_manager[0].scale
    pixel_separation = model_parameters.peak_separation/s_adf_scale
    atom_position_list_pca = get_peak2d_skimage(
            s_adf_modified,
            separation=pixel_separation)[0]

    #################################
    path_name = s_adf_filename
    path_name = path_name[0: path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    normalized_abf_data = 1./s_abf.data
    normalized_abf_data = normalized_abf_data-normalized_abf_data.min()
    normalized_abf_data = normalized_abf_data/normalized_abf_data.max()
    atom_lattice.inverted_abf_image = np.rot90(np.fliplr(normalized_abf_data))

    sublattice_0_param = model_parameters.get_sublattice_from_order(0)

    sublattice_0 = Sublattice(
            atom_position_list_pca,
            np.rot90(np.fliplr(s_adf_modified.data)))

    sublattice_0.save_path = "./" + path_name + "/"
    sublattice_0.path_name = path_name
    sublattice_0.plot_color = sublattice_0_param.color
    sublattice_0.name = sublattice_0_param.name
    sublattice_0.tag = sublattice_0_param.tag
    sublattice_0.pixel_size = s_adf.axes_manager[0].scale
    sublattice_0.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sublattice_list.append(sublattice_0)

    for atom in sublattice_0.atom_list:
        atom.sigma_x = 0.05/sublattice_0.pixel_size
        atom.sigma_y = 0.05/sublattice_0.pixel_size

    print("Refining " + sublattice_0.name)
    refine_sublattice(
            sublattice_0,
            [
                (sublattice_0.original_adf_image, 2, 'gaussian')],
            0.35)

    plt.close('all')
    construct_zone_axes_from_sublattice(sublattice_0)

    plt.ion()
    return(atom_lattice)


def run_process_for_adf_image_a_cation(
        s_adf_filename,
        peak_separation=0.30,
        filter_signal=True):
    plt.ioff()
    ################
    s_adf = hs.load(s_adf_filename)
    s_adf_modified = s_adf.deepcopy()
    if filter_signal:
        s_adf_modified = subtract_average_background(s_adf_modified)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
    peak_separation1 = peak_separation/s_adf.axes_manager[0].scale
    atom_position_list_pca = get_peak2d_skimage(
            s_adf_modified,
            separation=peak_separation1)[0]

    #################################
    path_name = s_adf_filename
    path_name = path_name[0: path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_sublattice = Sublattice(
            atom_position_list_pca,
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sublattice.save_path = "./" + path_name + "/"
    a_sublattice.path_name = path_name
    a_sublattice.plot_color = 'blue'
    a_sublattice.tag = 'a'
    a_sublattice.pixel_size = s_adf.axes_manager[0].scale
    a_sublattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sublattice_list.append(a_sublattice)

    for atom in a_sublattice.atom_list:
        atom.sigma_x = 0.05/a_sublattice.pixel_size
        atom.sigma_y = 0.05/a_sublattice.pixel_size

    a_sublattice.plot_atom_list_on_image_data(
            figname=a_sublattice.tag+"_atom_refine0_initial.jpg")

    print("Refining a atom lattice")
    refine_sublattice(
            a_sublattice,
            [
                (a_sublattice.adf_image, 1, 'center_of_mass')],
            0.50)
    refine_sublattice(
            a_sublattice,
            [
                (
                    a_sublattice.original_adf_image,
                    1,
                    'center_of_mass')],
            0.50)
    a_sublattice.plot_atom_list_on_image_data(
            figname=a_sublattice.tag+"_atom_refine1_com.jpg")
    refine_sublattice(
            a_sublattice,
            [
                (a_sublattice.original_adf_image, 1, 'gaussian')],
            0.50)
    a_sublattice.plot_atom_list_on_image_data(
            figname=a_sublattice.tag+"_atom_refine2_gaussian.jpg")
    atom_lattice.save_atom_lattice(
            filename=a_sublattice.save_path + "atom_lattice.hdf5")
    plt.close('all')
    construct_zone_axes_from_sublattice(a_sublattice)

    return(atom_lattice)


def run_peak_finding_process_for_all_datasets(
        refinement_interations=2):
    dm3_adf_filename_list = glob.glob("*ADF*.dm3")
    dm3_adf_filename_list.sort()
    dataset_list = []
    total_datasets = len(dm3_adf_filename_list)+1
    for index, dm3_adf_filename in enumerate(dm3_adf_filename_list):
        print(
                "Dataset " + str(index+1) +
                "/" + str(total_datasets) +
                ": " + dm3_adf_filename)
        dm3_abf_filename = dm3_adf_filename.replace("ADF", "ABF")
        dataset = run_peak_finding_process_for_single_dataset(
                dm3_adf_filename,
                dm3_abf_filename,
                refinement_interations=refinement_interations)
        dataset_list.append(dataset)
    return(dataset_list)


def run_peak_finding_process_for_single_dataset(
        s_adf_filename,
        s_abf_filename,
        peak_separation=0.13,  # in nanometers
        refinement_interation_config=None,
        invert_abf_signal=True):
    plt.ioff()
    ################
    s_abf = hs.load(s_abf_filename)
    s_abf.change_dtype('float64')
    s_abf_modified = subtract_average_background(s_abf)
    s_abf_modified = do_pca_on_signal(s_abf_modified)
    s_abf_modified = normalize_signal(
            s_abf_modified, invert_signal=invert_abf_signal)
    if invert_abf_signal:
        s_abf.data = 1./s_abf.data

    ################
    s_adf = hs.load(s_adf_filename)
    s_adf.change_dtype('float64')
    s_adf_modified = subtract_average_background(s_adf)
    s_adf_modified = do_pca_on_signal(s_adf_modified)

    pixel_separation = peak_separation/s_adf.axes_manager[0].scale
    atom_position_list_pca = get_peak2d_skimage(
            s_adf_modified,
            separation=pixel_separation)[0]

    #################################
    path_name = s_adf_filename
    path_name = path_name[0: path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    normalized_abf_data = 1./s_abf.data
    normalized_abf_data = normalized_abf_data-normalized_abf_data.min()
    normalized_abf_data = normalized_abf_data/normalized_abf_data.max()
    atom_lattice.inverted_abf_image = np.rot90(np.fliplr(normalized_abf_data))

    a_sublattice = Sublattice(
            atom_position_list_pca,
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sublattice.save_path = "./" + path_name + "/"
    a_sublattice.path_name = path_name
    a_sublattice.plot_color = 'blue'
    a_sublattice.tag = 'a'
    a_sublattice.pixel_size = s_adf.axes_manager[0].scale
    a_sublattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sublattice_list.append(a_sublattice)

    for atom in a_sublattice.atom_list:
        atom.sigma_x = 0.05/a_sublattice.pixel_size
        atom.sigma_y = 0.05/a_sublattice.pixel_size

    print("Refining a atom lattice")
    refine_sublattice(
            a_sublattice,
            [
                (a_sublattice.original_adf_image, 2, 'gaussian')],
            0.35)

    plt.close('all')
    construct_zone_axes_from_sublattice(a_sublattice)

    zone_vector_100 = a_sublattice.zones_axis_average_distances[1]
    b_atom_list = a_sublattice.find_missing_atoms_from_zone_vector(
            zone_vector_100, new_atom_tag='B')

    b_sublattice = Sublattice(
            b_atom_list,
            np.rot90(np.fliplr(s_adf_modified.data)))
    b_sublattice.save_path = "./" + path_name + "/"
    b_sublattice.path_name = path_name
    b_sublattice.plot_color = 'green'
    b_sublattice.tag = 'b'
    b_sublattice.pixel_size = s_adf.axes_manager[0].scale
    b_sublattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sublattice_list.append(b_sublattice)

    for atom in b_sublattice.atom_list:
        atom.sigma_x = 0.03/b_sublattice.pixel_size
        atom.sigma_y = 0.03/b_sublattice.pixel_size
    construct_zone_axes_from_sublattice(b_sublattice)
    image_atoms_removed = b_sublattice.original_adf_image
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed,
        a_sublattice,
        percent_to_nn=0.35)

    b_sublattice.original_adf_image_atoms_removed = image_atoms_removed
    b_sublattice.adf_image = image_atoms_removed

    print("Refining b atom lattice")
    refine_sublattice(
            b_sublattice,
            [
                (image_atoms_removed, 2, 'center_of_mass'),
                (image_atoms_removed, 2, 'gaussian')],
            0.25)

    plt.close('all')

    zone_vector_110 = b_sublattice.zones_axis_average_distances[0]
    o_atom_list = b_sublattice.find_missing_atoms_from_zone_vector(
            zone_vector_110, new_atom_tag='O')

    o_sublattice = Sublattice(
            o_atom_list, np.rot90(np.fliplr(s_abf_modified.data)))
    o_sublattice.save_path = "./" + path_name + "/"
    o_sublattice.path_name = path_name
    o_sublattice.plot_color = 'red'
    o_sublattice.tag = 'o'
    o_sublattice.pixel_size = s_abf.axes_manager[0].scale
    o_sublattice.original_adf_image = np.rot90(np.fliplr(s_abf.data))
#    o_sublattice.plot_clim = (0.0, 0.2)
    construct_zone_axes_from_sublattice(o_sublattice)
    image_atoms_removed = o_sublattice.original_adf_image
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed,
        a_sublattice,
        percent_to_nn=0.40)
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed,
        b_sublattice,
        percent_to_nn=0.30)

    o_sublattice.adf_image = image_atoms_removed

    for atom in o_sublattice.atom_list:
        atom.sigma_x = 0.025/b_sublattice.pixel_size
        atom.sigma_y = 0.025/b_sublattice.pixel_size
    atom_lattice.sublattice_list.append(o_sublattice)

    print("Refining o atom lattice")
    refine_sublattice(
            o_sublattice,
            [
                (image_atoms_removed, 1, 'center_of_mass'),
                (image_atoms_removed, 2, 'gaussian')],
            0.2)

    atom_lattice.save_atom_lattice()

    plt.close('all')
    plt.ion()
    return(atom_lattice)


def run_process_for_adf_image_a_cation(
        s_adf_filename,
        peak_separation=0.30,
        filter_signal=True):
    plt.ioff()
    ################
    s_adf = hs.load(s_adf_filename)
    s_adf_modified = s_adf.deepcopy()
    if filter_signal:
        s_adf_modified = subtract_average_background(s_adf_modified)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
    peak_separation1 = peak_separation/s_adf.axes_manager[0].scale
    atom_position_list_pca = get_peak2d_skimage(
            s_adf_modified,
            separation=peak_separation1)[0]

    #################################
    path_name = s_adf_filename
    path_name = path_name[0: path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_sublattice = Sublattice(
            atom_position_list_pca,
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sublattice.save_path = "./" + path_name + "/"
    a_sublattice.path_name = path_name
    a_sublattice.plot_color = 'blue'
    a_sublattice.tag = 'a'
    a_sublattice.pixel_size = s_adf.axes_manager[0].scale
    a_sublattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sublattice_list.append(a_sublattice)

    for atom in a_sublattice.atom_list:
        atom.sigma_x = 0.05/a_sublattice.pixel_size
        atom.sigma_y = 0.05/a_sublattice.pixel_size

    a_sublattice.plot_atom_list_on_image_data(
            figname=a_sublattice.tag+"_atom_refine0_initial.jpg")

    print("Refining a atom lattice")
    refine_sublattice(
            a_sublattice,
            [
                (a_sublattice.adf_image, 1, 'center_of_mass')],
            0.50)
    refine_sublattice(
            a_sublattice,
            [
                (
                    a_sublattice.original_adf_image,
                    1,
                    'center_of_mass')],
            0.50)
    a_sublattice.plot_atom_list_on_image_data(
            figname=a_sublattice.tag+"_atom_refine1_com.jpg")
    refine_sublattice(
            a_sublattice,
            [
                (a_sublattice.original_adf_image, 1, 'gaussian')],
            0.50)
    a_sublattice.plot_atom_list_on_image_data(
            figname=a_sublattice.tag+"_atom_refine2_gaussian.jpg")
    atom_lattice.save_atom_lattice(
            filename=a_sublattice.save_path +
            "atom_lattice.hdf5")
    plt.close('all')
    construct_zone_axes_from_sublattice(a_sublattice)

    return(atom_lattice)


def run_process_for_adf_image_a_b_cation(
        s_adf_filename,
        peak_separation=0.13,
        filter_signal=True):
    plt.ioff()
    ################
    s_adf = hs.load(s_adf_filename)
    s_adf_modified = s_adf.deepcopy()
    if filter_signal:
        s_adf_modified = subtract_average_background(s_adf_modified)
        s_adf_modified = do_pca_on_signal(s_adf_modified)
    peak_separation1 = peak_separation/s_adf.axes_manager[0].scale
    atom_position_list_pca = get_peak2d_skimage(
            s_adf_modified,
            separation=peak_separation1)[0]

    #################################
    path_name = s_adf_filename
    path_name = path_name[0: path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_sublattice = Sublattice(
            atom_position_list_pca,
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sublattice.save_path = "./" + path_name + "/"
    a_sublattice.path_name = path_name
    a_sublattice.plot_color = 'blue'
    a_sublattice.tag = 'a'
    a_sublattice.pixel_size = s_adf.axes_manager[0].scale
    a_sublattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sublattice_list.append(a_sublattice)

    for atom in a_sublattice.atom_list:
        atom.sigma_x = 0.05/a_sublattice.pixel_size
        atom.sigma_y = 0.05/a_sublattice.pixel_size

    atom_lattice.save_atom_lattice(
            filename=a_sublattice.save_path +
            "atom_lattice_no_refinement.hdf5")
    print("Refining a atom lattice")
    refine_sublattice(
            a_sublattice,
            [
                (a_sublattice.adf_image, 1, 'center_of_mass')],
            0.25)
    refine_sublattice(
            a_sublattice,
            [
                (
                    a_sublattice.original_adf_image,
                    1,
                    'center_of_mass')],
            0.30)
    atom_lattice.save_atom_lattice(
            filename=a_sublattice.save_path +
            "atom_lattice_center_of_mass.hdf5")
    refine_sublattice(
            a_sublattice,
            [
                (a_sublattice.original_adf_image, 1, 'gaussian')],
            0.30)
    atom_lattice.save_atom_lattice(
            filename=a_sublattice.save_path +
            "atom_lattice_2d_model.hdf5")
    plt.close('all')
    construct_zone_axes_from_sublattice(a_sublattice)

#    zone_vector_100 = a_sublattice.zones_axis_average_distances[1]
#    b_atom_list = a_sublattice.find_missing_atoms_from_zone_vector(
#            zone_vector_100, new_atom_tag='B')
#
#    b_sublattice = Sublattice(
#        b_atom_list,
#        np.rot90(np.fliplr(s_adf_modified.data)))
#    atom_lattice.sublattice_list.append(b_sublattice)
#    b_sublattice.save_path = "./" + path_name + "/"
#    b_sublattice.path_name = path_name
#    b_sublattice.plot_color = 'green'
#    b_sublattice.tag = 'b'
#    b_sublattice.pixel_size = s_adf.axes_manager[0].scale
#    b_sublattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
#
#    for atom in b_sublattice.atom_list:
#        atom.sigma_x = 0.03/b_sublattice.pixel_size
#        atom.sigma_y = 0.03/b_sublattice.pixel_size
#
#    image_atoms_removed = b_sublattice.original_adf_image
#    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
#        image_atoms_removed,
#        a_sublattice,
#        percent_to_nn=0.35)
#    construct_zone_axes_from_sublattice(b_sublattice)
#
#    b_sublattice.adf_image = image_atoms_removed
#
#    print("Refining b atom lattice")
#    refine_sublattice(
#            b_sublattice,
#            [
#                (b_sublattice.adf_image, 2, 'gaussian')],
#            0.3)
#
#
#    plt.close('all')
    plt.ion()
    return(atom_lattice)
