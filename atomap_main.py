import os
import glob
import matplotlib.pyplot as plt
import hyperspy.api as hs
import numpy as np
from atomap_atom_finding_refining import\
        subtract_average_background,\
        do_pca_on_signal,\
        refine_sub_lattice,\
        construct_zone_axes_from_sub_lattice,\
        get_peak2d_skimage,\
        normalize_signal

from atomap_tools import\
        remove_atoms_from_image_using_2d_gaussian

from atom_lattice_class import Atom_Lattice
from sub_lattice_class import Sub_Lattice

class SubLatticeParameterBase:
    def __init__(self):
        self.color = 'red'
        self.name = "Base Sublattice"
        self.sublattice_order = None

    def __repr__(self):
        return '<%s, %s>' % (
            self.__class__.__name__, 
            self.name
            )

class PerovskiteOxide110SubLatticeACation(SubLatticeParameterBase):
    def __init__(self):
        SubLatticeParameterBase.__init__(self)
        self.name = "A-cation"
        self.tag = "A"
        self.color = 'blue'
        self.zone_axis = [
                {'number':0, 'name':'110'},
                {'number':1, 'name':'100'},
                {'number':2, 'name':'11-2'},
                {'number':3, 'name':'112'},
                {'number':5, 'name':'111'},
                {'number':6, 'name':'11-1'},
                ]
        self.sublattice_order = 0

class PerovskiteOxide110SubLatticeBCation(SubLatticeParameterBase):
    def __init__(self):
        SubLatticeParameterBase.__init__(self)
        self.name = "B-cation"
        self.tag = "B"
        self.color = 'green'
        self.zone_axis = [
                {'number':0, 'name':'110'},
                {'number':1, 'name':'100'},
                {'number':2, 'name':'11-2'},
                {'number':3, 'name':'112'},
                {'number':5, 'name':'111'},
                {'number':6, 'name':'11-1'},]
        self.sublattice_order = 1
        self.sublattice_position_sublattice = "A-cation"
        self.sublattice_position_zoneaxis = "100"

class ModelParameters:
    def __init__(self):
        self.peak_separation = None
        self.number_of_sublattices = None
        self.name = None
    
    def __repr__(self):
        return '<%s, %s>' % (
            self.__class__.__name__, 
            self.name,
            )

class PerovskiteOxide110(ModelParameters):
    def __init__(self):
        ModelParameters.__init__(self)
        self.name = "Peroskite 110"
        self.peak_separation = 0.127
        self.number_of_sublattices = 1

        self.sublattice_list = [
            PerovskiteOxide110SubLatticeACation(),
            ]

    def get_sublattice_from_order(self, order_number):
        for sublattice in self.sublattice_list:
            if order_number == sublattice.sublattice_order:
                return(sublattice)

class SrTiO3_110(PerovskiteOxide110):
    def __init__(self):
        PerovskiteOxide110.__init__(self)
        self.number_of_sublattices = 3
        self.sublattice_names = "Sr", "Ti", "O"
        Ti_sublattice_position = {
                "sublattice":"Sr",
                "zoneaxis":"100"}
        O_sublattice_position = {
                "sublattice":"Ti",
                "zoneaxis":"110"}
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
        image1_filename=None):
    image0 = hs.load(image0_filename)
    image0_modified = run_image_filtering(image0)

    if model_parameters is None:
        model_parameters = PerovskiteOxide110()

    pixel_separation = model_parameters.peak_separation/image0.axes_manager[0].scale
    initial_atom_position_list = get_peak2d_skimage(
            image0_modified, 
            separation=pixel_separation)[0]

    #################################
    path_name = image0_filename
    path_name = path_name[0:path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    
    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = image0_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(image0.data))

    ### WORK IN PROGRESS

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

    pixel_separation = model_parameters.peak_separation/s_adf.axes_manager[0].scale
    atom_position_list_pca = get_peak2d_skimage(
            s_adf_modified, 
            separation=pixel_separation)[0]

    #################################
    path_name = s_adf_filename
    path_name = path_name[0:path_name.rfind(".")]
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

    sub_lattice_0 = Sub_Lattice(
            atom_position_list_pca, 
            np.rot90(np.fliplr(s_adf_modified.data)))

    sub_lattice_0.save_path = "./" + path_name + "/"
    sub_lattice_0.path_name = path_name
    sub_lattice_0.plot_color = sublattice_0_param.color
    sub_lattice_0.name = sublattice_0_param.name
    sub_lattice_0.tag = sublattice_0_param.tag
    sub_lattice_0.pixel_size = s_adf.axes_manager[0].scale
    sub_lattice_0.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sub_lattice_list.append(sub_lattice_0)

    for atom in sub_lattice_0.atom_list:
        atom.sigma_x = 0.05/sub_lattice_0.pixel_size
        atom.sigma_y = 0.05/sub_lattice_0.pixel_size

    print("Refining " + sub_lattice_0.name)
    refine_sub_lattice(
            sub_lattice_0, 
            [
#                (sub_lattice_0.adf_image, 2, 'gaussian'),
                (sub_lattice_0.original_adf_image, 2, 'gaussian')],
            0.35)

    plt.close('all')
    construct_zone_axes_from_sub_lattice(sub_lattice_0)

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
    path_name = path_name[0:path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_sub_lattice = Sub_Lattice(
            atom_position_list_pca, 
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sub_lattice.save_path = "./" + path_name + "/"
    a_sub_lattice.path_name = path_name
    a_sub_lattice.plot_color = 'blue'
    a_sub_lattice.tag = 'a'
    a_sub_lattice.pixel_size = s_adf.axes_manager[0].scale
    a_sub_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sub_lattice_list.append(a_sub_lattice)

    for atom in a_sub_lattice.atom_list:
        atom.sigma_x = 0.05/a_sub_lattice.pixel_size
        atom.sigma_y = 0.05/a_sub_lattice.pixel_size

    a_sub_lattice.plot_atom_list_on_image_data(
            figname=a_sub_lattice.tag+"_atom_refine0_initial.jpg")

    print("Refining a atom lattice")
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (a_sub_lattice.adf_image, 1, 'center_of_mass')],
            0.50)
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (
                    a_sub_lattice.original_adf_image,
                    1, 
                    'center_of_mass')],
            0.50)
    a_sub_lattice.plot_atom_list_on_image_data(
            figname=a_sub_lattice.tag+"_atom_refine1_com.jpg")
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (a_sub_lattice.original_adf_image, 1, 'gaussian')],
            0.50)
    a_sub_lattice.plot_atom_list_on_image_data(
            figname=a_sub_lattice.tag+"_atom_refine2_gaussian.jpg")
    atom_lattice.save_atom_lattice(
            filename=a_sub_lattice.save_path +\
                    "atom_lattice.hdf5")
    plt.close('all')
    construct_zone_axes_from_sub_lattice(a_sub_lattice)

    return(atom_lattice)

def run_peak_finding_process_for_all_datasets(
        refinement_interations=2):
    dm3_adf_filename_list = glob.glob("*ADF*.dm3")    
    dm3_adf_filename_list.sort()
    dataset_list = []
    total_datasets = len(dm3_adf_filename_list)+1
    for index, dm3_adf_filename in enumerate(dm3_adf_filename_list):
        print(
                "Dataset "+str(index+1)+\
                "/"+str(total_datasets)+\
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
        peak_separation=0.13, # in nanometers
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
    path_name = path_name[0:path_name.rfind(".")]
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

    a_sub_lattice = Sub_Lattice(
            atom_position_list_pca, 
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sub_lattice.save_path = "./" + path_name + "/"
    a_sub_lattice.path_name = path_name
    a_sub_lattice.plot_color = 'blue'
    a_sub_lattice.tag = 'a'
    a_sub_lattice.pixel_size = s_adf.axes_manager[0].scale
    a_sub_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sub_lattice_list.append(a_sub_lattice)

    for atom in a_sub_lattice.atom_list:
        atom.sigma_x = 0.05/a_sub_lattice.pixel_size
        atom.sigma_y = 0.05/a_sub_lattice.pixel_size

    print("Refining a atom lattice")
    refine_sub_lattice(
            a_sub_lattice, 
            [
#                (a_sub_lattice.adf_image, 2, 'gaussian'),
                (a_sub_lattice.original_adf_image, 2, 'gaussian')],
            0.35)

    plt.close('all')
    construct_zone_axes_from_sub_lattice(a_sub_lattice)

    zone_vector_100 = a_sub_lattice.zones_axis_average_distances[1]
    b_atom_list = a_sub_lattice.find_missing_atoms_from_zone_vector(
            zone_vector_100, new_atom_tag='B')

    b_sub_lattice = Sub_Lattice(
            b_atom_list, 
            np.rot90(np.fliplr(s_adf_modified.data)))
    b_sub_lattice.save_path = "./" + path_name + "/"
    b_sub_lattice.path_name = path_name
    b_sub_lattice.plot_color = 'green'
    b_sub_lattice.tag = 'b'
    b_sub_lattice.pixel_size = s_adf.axes_manager[0].scale
    b_sub_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sub_lattice_list.append(b_sub_lattice)

    for atom in b_sub_lattice.atom_list:
        atom.sigma_x = 0.03/b_sub_lattice.pixel_size
        atom.sigma_y = 0.03/b_sub_lattice.pixel_size
    construct_zone_axes_from_sub_lattice(b_sub_lattice)
    image_atoms_removed = b_sub_lattice.original_adf_image
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed, 
        a_sub_lattice,
        percent_distance_to_nearest_neighbor=0.35)

    b_sub_lattice.original_adf_image_atoms_removed =\
            image_atoms_removed
    b_sub_lattice.adf_image = image_atoms_removed

    print("Refining b atom lattice")
    refine_sub_lattice(
            b_sub_lattice, 
            [
                (image_atoms_removed, 2, 'center_of_mass'),
                (image_atoms_removed, 2, 'gaussian')],
            0.25)

    plt.close('all')

    zone_vector_110 = b_sub_lattice.zones_axis_average_distances[0]
    o_atom_list = b_sub_lattice.find_missing_atoms_from_zone_vector(
            zone_vector_110, new_atom_tag='O')

    o_sub_lattice = Sub_Lattice(
            o_atom_list, np.rot90(np.fliplr(s_abf_modified.data)))
    o_sub_lattice.save_path = "./" + path_name + "/"
    o_sub_lattice.path_name = path_name
    o_sub_lattice.plot_color = 'red'
    o_sub_lattice.tag = 'o'
    o_sub_lattice.pixel_size = s_abf.axes_manager[0].scale
    o_sub_lattice.original_adf_image = np.rot90(np.fliplr(s_abf.data))
#    o_sub_lattice.plot_clim = (0.0, 0.2)
    construct_zone_axes_from_sub_lattice(o_sub_lattice)
    image_atoms_removed = o_sub_lattice.original_adf_image
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed, 
        a_sub_lattice,
        percent_distance_to_nearest_neighbor=0.40)
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed, 
        b_sub_lattice,
        percent_distance_to_nearest_neighbor=0.30)

    o_sub_lattice.adf_image = image_atoms_removed

    for atom in o_sub_lattice.atom_list:
        atom.sigma_x = 0.025/b_sub_lattice.pixel_size
        atom.sigma_y = 0.025/b_sub_lattice.pixel_size
    atom_lattice.sub_lattice_list.append(o_sub_lattice)

    print("Refining o atom lattice")
    refine_sub_lattice(
            o_sub_lattice, 
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
    path_name = path_name[0:path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_sub_lattice = Sub_Lattice(
            atom_position_list_pca, 
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sub_lattice.save_path = "./" + path_name + "/"
    a_sub_lattice.path_name = path_name
    a_sub_lattice.plot_color = 'blue'
    a_sub_lattice.tag = 'a'
    a_sub_lattice.pixel_size = s_adf.axes_manager[0].scale
    a_sub_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sub_lattice_list.append(a_sub_lattice)

    for atom in a_sub_lattice.atom_list:
        atom.sigma_x = 0.05/a_sub_lattice.pixel_size
        atom.sigma_y = 0.05/a_sub_lattice.pixel_size

    a_sub_lattice.plot_atom_list_on_image_data(
            figname=a_sub_lattice.tag+"_atom_refine0_initial.jpg")

    print("Refining a atom lattice")
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (a_sub_lattice.adf_image, 1, 'center_of_mass')],
            0.50)
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (
                    a_sub_lattice.original_adf_image,
                    1, 
                    'center_of_mass')],
            0.50)
    a_sub_lattice.plot_atom_list_on_image_data(
            figname=a_sub_lattice.tag+"_atom_refine1_com.jpg")
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (a_sub_lattice.original_adf_image, 1, 'gaussian')],
            0.50)
    a_sub_lattice.plot_atom_list_on_image_data(
            figname=a_sub_lattice.tag+"_atom_refine2_gaussian.jpg")
    atom_lattice.save_atom_lattice(
            filename=a_sub_lattice.save_path +\
                    "atom_lattice.hdf5")
    plt.close('all')
    construct_zone_axes_from_sub_lattice(a_sub_lattice)

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
    path_name = path_name[0:path_name.rfind(".")]
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    #########################################

    atom_lattice = Atom_Lattice()
    atom_lattice.original_filename = s_adf_filename
    atom_lattice.path_name = path_name
    atom_lattice.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_sub_lattice = Sub_Lattice(
            atom_position_list_pca, 
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_sub_lattice.save_path = "./" + path_name + "/"
    a_sub_lattice.path_name = path_name
    a_sub_lattice.plot_color = 'blue'
    a_sub_lattice.tag = 'a'
    a_sub_lattice.pixel_size = s_adf.axes_manager[0].scale
    a_sub_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    atom_lattice.sub_lattice_list.append(a_sub_lattice)

    for atom in a_sub_lattice.atom_list:
        atom.sigma_x = 0.05/a_sub_lattice.pixel_size
        atom.sigma_y = 0.05/a_sub_lattice.pixel_size

    atom_lattice.save_atom_lattice(
            filename=a_sub_lattice.save_path +\
                    "atom_lattice_no_refinement.hdf5")
    print("Refining a atom lattice")
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (a_sub_lattice.adf_image, 1, 'center_of_mass')],
            0.25)
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (
                    a_sub_lattice.original_adf_image, 
                    1, 
                    'center_of_mass')],
            0.30)
    atom_lattice.save_atom_lattice(
            filename=a_sub_lattice.save_path +\
                    "atom_lattice_center_of_mass.hdf5")
    refine_sub_lattice(
            a_sub_lattice, 
            [
                (a_sub_lattice.original_adf_image, 1, 'gaussian')],
            0.30)
    atom_lattice.save_atom_lattice(
            filename=a_sub_lattice.save_path +\
                    "atom_lattice_2d_model.hdf5")
    plt.close('all')
    construct_zone_axes_from_sub_lattice(a_sub_lattice)

#    zone_vector_100 = a_sub_lattice.zones_axis_average_distances[1]
#    b_atom_list = a_sub_lattice.find_missing_atoms_from_zone_vector(
#            zone_vector_100, new_atom_tag='B')
#
#    b_sub_lattice = Sub_Lattice(b_atom_list, np.rot90(np.fliplr(s_adf_modified.data)))
#    atom_lattice.sub_lattice_list.append(b_sub_lattice)
#    b_sub_lattice.save_path = "./" + path_name + "/"
#    b_sub_lattice.path_name = path_name
#    b_sub_lattice.plot_color = 'green'
#    b_sub_lattice.tag = 'b'
#    b_sub_lattice.pixel_size = s_adf.axes_manager[0].scale
#    b_sub_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
#
#    for atom in b_sub_lattice.atom_list:
#        atom.sigma_x = 0.03/b_sub_lattice.pixel_size
#        atom.sigma_y = 0.03/b_sub_lattice.pixel_size
#
#    image_atoms_removed = b_sub_lattice.original_adf_image
#    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
#        image_atoms_removed, 
#        a_sub_lattice,
#        percent_distance_to_nearest_neighbor=0.35)
#    construct_zone_axes_from_sub_lattice(b_sub_lattice)
#
#    b_sub_lattice.adf_image = image_atoms_removed
#
#    print("Refining b atom lattice")
#    refine_sub_lattice(
#            b_sub_lattice, 
#            [
#                (b_sub_lattice.adf_image, 2, 'gaussian')],
#            0.3)
#
#
#    plt.close('all')
    plt.ion()
    return(atom_lattice)
