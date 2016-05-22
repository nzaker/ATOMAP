import sys; sys.dont_write_bytecode = True
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import math
import operator
import copy
from scipy import ndimage
from scipy import interpolate
from matplotlib.gridspec import GridSpec
import os
import glob
import math
import json
from skimage.feature import peak_local_max
from scipy.stats import linregress
import h5py

from atomap_plotting import *

def run_peak_finding_process_for_all_datasets(refinement_interations=2):
    dm3_adf_filename_list = glob.glob("*ADF*.dm3")    
    dm3_adf_filename_list.sort()
    dataset_list = []
    total_datasets = len(dm3_adf_filename_list)+1
    for index, dm3_adf_filename in enumerate(dm3_adf_filename_list):
        print("Dataset "+str(index+1)+"/"+str(total_datasets)+": " + dm3_adf_filename)
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

    material_structure = Material_Structure()
    material_structure.original_filename = s_adf_filename
    material_structure.path_name = path_name
    material_structure.adf_image = np.rot90(np.fliplr(s_adf.data))

    normalized_abf_data = 1./s_abf.data
    normalized_abf_data = normalized_abf_data-normalized_abf_data.min()
    normalized_abf_data = normalized_abf_data/normalized_abf_data.max()
    material_structure.inverted_abf_image = np.rot90(np.fliplr(normalized_abf_data))

    a_atom_lattice = Atom_Lattice(
            atom_position_list_pca, np.rot90(np.fliplr(s_adf_modified.data)))

    a_atom_lattice.save_path = "./" + path_name + "/"
    a_atom_lattice.path_name = path_name
    a_atom_lattice.plot_color = 'blue'
    a_atom_lattice.tag = 'a'
    a_atom_lattice.pixel_size = s_adf.axes_manager[0].scale
    a_atom_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    material_structure.atom_lattice_list.append(a_atom_lattice)

    for atom in a_atom_lattice.atom_list:
        atom.sigma_x = 0.05/a_atom_lattice.pixel_size
        atom.sigma_y = 0.05/a_atom_lattice.pixel_size

    print("Refining a atom lattice")
    refine_atom_lattice(
            a_atom_lattice, 
            [
#                (a_atom_lattice.adf_image, 2, 'gaussian'),
                (a_atom_lattice.original_adf_image, 2, 'gaussian')],
            0.35)

    plt.close('all')
    construct_zone_axes_from_atom_lattice(a_atom_lattice)

    zone_vector_100 = a_atom_lattice.zones_axis_average_distances[1]
    b_atom_list = a_atom_lattice.find_missing_atoms_from_zone_vector(
            zone_vector_100, new_atom_tag='B')

    b_atom_lattice = Atom_Lattice(b_atom_list, np.rot90(np.fliplr(s_adf_modified.data)))
    b_atom_lattice.save_path = "./" + path_name + "/"
    b_atom_lattice.path_name = path_name
    b_atom_lattice.plot_color = 'green'
    b_atom_lattice.tag = 'b'
    b_atom_lattice.pixel_size = s_adf.axes_manager[0].scale
    b_atom_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    material_structure.atom_lattice_list.append(b_atom_lattice)

    for atom in b_atom_lattice.atom_list:
        atom.sigma_x = 0.03/b_atom_lattice.pixel_size
        atom.sigma_y = 0.03/b_atom_lattice.pixel_size
    construct_zone_axes_from_atom_lattice(b_atom_lattice)
    image_atoms_removed = b_atom_lattice.original_adf_image
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed, 
        a_atom_lattice,
        percent_distance_to_nearest_neighbor=0.35)

    b_atom_lattice.original_adf_image_atoms_removed = image_atoms_removed
    b_atom_lattice.adf_image = image_atoms_removed

    print("Refining b atom lattice")
    refine_atom_lattice(
            b_atom_lattice, 
            [
                (image_atoms_removed, 2, 'center_of_mass'),
                (image_atoms_removed, 2, 'gaussian')],
            0.25)

    plt.close('all')

    zone_vector_110 = b_atom_lattice.zones_axis_average_distances[0]
    o_atom_list = b_atom_lattice.find_missing_atoms_from_zone_vector(
            zone_vector_110, new_atom_tag='O')

    o_atom_lattice = Atom_Lattice(o_atom_list, np.rot90(np.fliplr(s_abf_modified.data)))
    o_atom_lattice.save_path = "./" + path_name + "/"
    o_atom_lattice.path_name = path_name
    o_atom_lattice.plot_color = 'red'
    o_atom_lattice.tag = 'o'
    o_atom_lattice.pixel_size = s_abf.axes_manager[0].scale
    o_atom_lattice.original_adf_image = np.rot90(np.fliplr(s_abf.data))
#    o_atom_lattice.plot_clim = (0.0, 0.2)
    construct_zone_axes_from_atom_lattice(o_atom_lattice)
    image_atoms_removed = o_atom_lattice.original_adf_image
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed, 
        a_atom_lattice,
        percent_distance_to_nearest_neighbor=0.40)
    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
        image_atoms_removed, 
        b_atom_lattice,
        percent_distance_to_nearest_neighbor=0.30)

    o_atom_lattice.adf_image = image_atoms_removed

    for atom in o_atom_lattice.atom_list:
        atom.sigma_x = 0.025/b_atom_lattice.pixel_size
        atom.sigma_y = 0.025/b_atom_lattice.pixel_size
    material_structure.atom_lattice_list.append(o_atom_lattice)

    print("Refining o atom lattice")
    refine_atom_lattice(
            o_atom_lattice, 
            [
                (image_atoms_removed, 1, 'center_of_mass'),
                (image_atoms_removed, 2, 'gaussian')],
            0.2)

    save_material_structure(material_structure)

    plt.close('all')
    plt.ion()
    return(material_structure)

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

    material_structure = Material_Structure()
    material_structure.original_filename = s_adf_filename
    material_structure.path_name = path_name
    material_structure.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_atom_lattice = Atom_Lattice(
            atom_position_list_pca, 
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_atom_lattice.save_path = "./" + path_name + "/"
    a_atom_lattice.path_name = path_name
    a_atom_lattice.plot_color = 'blue'
    a_atom_lattice.tag = 'a'
    a_atom_lattice.pixel_size = s_adf.axes_manager[0].scale
    a_atom_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    material_structure.atom_lattice_list.append(a_atom_lattice)

    for atom in a_atom_lattice.atom_list:
        atom.sigma_x = 0.05/a_atom_lattice.pixel_size
        atom.sigma_y = 0.05/a_atom_lattice.pixel_size

    a_atom_lattice.plot_atom_list_on_stem_data(
            figname=a_atom_lattice.tag+"_atom_refine0_initial.jpg")

    print("Refining a atom lattice")
    refine_atom_lattice(
            a_atom_lattice, 
            [
                (a_atom_lattice.adf_image, 1, 'center_of_mass')],
            0.50)
    refine_atom_lattice(
            a_atom_lattice, 
            [
                (a_atom_lattice.original_adf_image, 1, 'center_of_mass')],
            0.50)
    a_atom_lattice.plot_atom_list_on_stem_data(
            figname=a_atom_lattice.tag+"_atom_refine1_com.jpg")
    refine_atom_lattice(
            a_atom_lattice, 
            [
                (a_atom_lattice.original_adf_image, 1, 'gaussian')],
            0.50)
    a_atom_lattice.plot_atom_list_on_stem_data(
            figname=a_atom_lattice.tag+"_atom_refine2_gaussian.jpg")
    save_material_structure(
            material_structure, 
            filename=a_atom_lattice.save_path + "material_structure.hdf5")
    plt.close('all')
    construct_zone_axes_from_atom_lattice(a_atom_lattice)

    return(material_structure)

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

    material_structure = Material_Structure()
    material_structure.original_filename = s_adf_filename
    material_structure.path_name = path_name
    material_structure.adf_image = np.rot90(np.fliplr(s_adf.data))

    a_atom_lattice = Atom_Lattice(
            atom_position_list_pca, 
            np.rot90(np.fliplr(s_adf_modified.data)))

    a_atom_lattice.save_path = "./" + path_name + "/"
    a_atom_lattice.path_name = path_name
    a_atom_lattice.plot_color = 'blue'
    a_atom_lattice.tag = 'a'
    a_atom_lattice.pixel_size = s_adf.axes_manager[0].scale
    a_atom_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
    material_structure.atom_lattice_list.append(a_atom_lattice)

    for atom in a_atom_lattice.atom_list:
        atom.sigma_x = 0.05/a_atom_lattice.pixel_size
        atom.sigma_y = 0.05/a_atom_lattice.pixel_size

    save_material_structure(
            material_structure, 
            filename=a_atom_lattice.save_path + "material_structure_no_refinement.hdf5")
    print("Refining a atom lattice")
    refine_atom_lattice(
            a_atom_lattice, 
            [
                (a_atom_lattice.adf_image, 1, 'center_of_mass')],
            0.25)
    refine_atom_lattice(
            a_atom_lattice, 
            [
                (a_atom_lattice.original_adf_image, 1, 'center_of_mass')],
            0.30)
    save_material_structure(
            material_structure, 
            filename=a_atom_lattice.save_path + "material_structure_center_of_mass.hdf5")
    refine_atom_lattice(
            a_atom_lattice, 
            [
                (a_atom_lattice.original_adf_image, 1, 'gaussian')],
            0.30)
    save_material_structure(
            material_structure, 
            filename=a_atom_lattice.save_path + "material_structure_2d_model.hdf5")
    plt.close('all')
    construct_zone_axes_from_atom_lattice(a_atom_lattice)

#    zone_vector_100 = a_atom_lattice.zones_axis_average_distances[1]
#    b_atom_list = a_atom_lattice.find_missing_atoms_from_zone_vector(
#            zone_vector_100, new_atom_tag='B')
#
#    b_atom_lattice = Atom_Lattice(b_atom_list, np.rot90(np.fliplr(s_adf_modified.data)))
#    material_structure.atom_lattice_list.append(b_atom_lattice)
#    b_atom_lattice.save_path = "./" + path_name + "/"
#    b_atom_lattice.path_name = path_name
#    b_atom_lattice.plot_color = 'green'
#    b_atom_lattice.tag = 'b'
#    b_atom_lattice.pixel_size = s_adf.axes_manager[0].scale
#    b_atom_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
#
#    for atom in b_atom_lattice.atom_list:
#        atom.sigma_x = 0.03/b_atom_lattice.pixel_size
#        atom.sigma_y = 0.03/b_atom_lattice.pixel_size
#
#    image_atoms_removed = b_atom_lattice.original_adf_image
#    image_atoms_removed = remove_atoms_from_image_using_2d_gaussian(
#        image_atoms_removed, 
#        a_atom_lattice,
#        percent_distance_to_nearest_neighbor=0.35)
#    construct_zone_axes_from_atom_lattice(b_atom_lattice)
#
#    b_atom_lattice.adf_image = image_atoms_removed
#
#    print("Refining b atom lattice")
#    refine_atom_lattice(
#            b_atom_lattice, 
#            [
#                (b_atom_lattice.adf_image, 2, 'gaussian')],
#            0.3)
#
#
#    plt.close('all')
    plt.ion()
    return(material_structure)
