import sys; sys.dont_write_bytecode = True
import hyperspy.hspy as hspy
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
from hyperspy.model import Model2D
from scipy.stats import linregress
import h5py

# Move to materials structure class
def save_material_structure(self, filename=None):
    if filename == None:
        path = self.path_name
        filename = path + "/" + "material_structure.hdf5"

    h5f = h5py.File(filename, 'w')
    for atom_lattice in self.atom_lattice_list:
        subgroup_name = atom_lattice.tag + "_atom_lattice"
        modified_image_data = atom_lattice.adf_image
        original_image_data = atom_lattice.original_adf_image
        atom_positions = np.array(atom_lattice._get_atom_position_list())

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
        
        h5f[subgroup_name].attrs['pixel_size'] = atom_lattice.pixel_size
        h5f[subgroup_name].attrs['tag'] = atom_lattice.tag
        h5f[subgroup_name].attrs['path_name'] = atom_lattice.path_name
        h5f[subgroup_name].attrs['save_path'] = atom_lattice.save_path
        h5f[subgroup_name].attrs['plot_color'] = atom_lattice.plot_color

    h5f.create_dataset(
        "image_data0",
        data=self.adf_image,
        chunks=True,
        compression='gzip')
    h5f.attrs['path_name'] = self.path_name

    h5f.close()

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

            atom_lattice.pixel_size = atom_lattice_set.attrs['pixel_size']
            atom_lattice.tag = atom_lattice_set.attrs['tag']
            atom_lattice.path_name = atom_lattice_set.attrs['path_name']
            atom_lattice.save_path = atom_lattice_set.attrs['save_path']
            atom_lattice.plot_color = atom_lattice_set.attrs['plot_color']

            material_structure.atom_lattice_list.append(atom_lattice)

        if group_name == 'image_data0':
            material_structure.adf_image = h5f[group_name][:]

    material_structure.path_name = h5f.attrs['path_name']

    h5f.close()

    if construct_zone_axes:
        material_structure.construct_zone_axes_for_atom_lattices()
    return(material_structure)

# Move to Atom_Lattice class
def _to_dict(self):
    position_array = np.array(self._get_atom_position_list())
    metadata = {
            'tag': self.tag,
            'pixel_size': self.pixel_size,
            'path_name': self.path_name,
            'plot_color': self.plot_color,
            }
    modified_image_data = self.adf_image
    original_image_data = self.original_adf_image

# Remove atom from image using 2d gaussian model
def remove_atoms_from_image_using_2d_gaussian(
        image, atom_lattice,
        percent_distance_to_nearest_neighbor=0.40):
    model_image = np.zeros(image.shape)
    X,Y = np.meshgrid(np.arange(model_image.shape[1]), np.arange(model_image.shape[0]))
    for atom in atom_lattice.atom_list:
        percent_distance = percent_distance_to_nearest_neighbor
        for i in range(10):
            g = atom.fit_2d_gaussian_with_mask(
                    image,
                    rotation_enabled=True,
                    percent_distance_to_nearest_neighbor=percent_distance)
            if g == False:
                if i == 9:
                    break
                else:
                    percent_distance *= 0.95
            else:
                model_image += g.function(X,Y)
                break
    subtracted_image = copy.deepcopy(image) - model_image
    return(subtracted_image)

# Bytte navn etterhvert
def plot_vector_field(x_pos_list, y_pos_list, x_rot_list, y_rot_list):
    fig, ax = plt.subplots()
    ax.quiver(
            x_pos_list,
            y_pos_list,
            x_rot_list,
            y_rot_list,
            scale=20.0,
            headwidth=0.0,
            headlength=0.0,
            headaxislength=0.0,
            pivot='middle')
    ax.set_xlim(min(x_pos_list), max(x_pos_list))
    ax.set_ylim(min(y_pos_list), max(y_pos_list))
    fig.savefig("vector_field.png", dpi=400)

##### DENNE SKAL FLYTTES ETTERHVERT
def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    y,x = np.ogrid[-centerX:imageSizeX-centerX, -centerY:imageSizeY-centerY]
    mask = x*x + y*y <= radius*radius
    return(mask)

def get_atom_rows_square(
        atom_lattice, atom_row1, atom_row2, 
        interface_atom_row, zone_vector, debug_plot=False):
    ort_atom_row1, ort_atom_row2 = atom_row1.get_side_edge_atom_rows_between_self_and_another_atom_row(
            atom_row2, zone_vector)
    
    if debug_plot:
        atom_lattice.plot_atom_row_on_stem_data(
                [atom_row1, atom_row2, ort_atom_row1, ort_atom_row2],
                figname="atom_row_square_debug.jpg")

    atom_list = atom_lattice.get_atom_list_between_four_atom_rows(
            atom_row1, atom_row2, ort_atom_row1, ort_atom_row2)

    if debug_plot:
        atom_lattice.plot_atom_list_on_stem_data(
                atom_list, figname="atom_row_square_atom_list_debug.jpg")
    
    x_pos_list = []
    y_pos_list = []
    z_pos_list = []
    for atom in atom_list:
        x_pos_list.append(atom.pixel_x)
        y_pos_list.append(atom.pixel_y)        
        z_pos_list.append(0) 

    data_list = np.array([x_pos_list, y_pos_list, z_pos_list]).swapaxes(0,1)
    atom_layer_list = find_atom_position_1d_from_distance_list_and_atom_row(
            data_list, interface_atom_row, rebin_data=True)

    atom_layer_list = np.array(atom_layer_list)[:,0]
    x_pos_list = []
    z_pos_list = []
    for index, atom_layer_pos in enumerate(atom_layer_list):
        if not (index == 0):
            previous_atom_layer = atom_layer_list[index-1]
            x_pos_list.append(0.5*(
                        atom_layer_pos+
                        previous_atom_layer))
            z_pos_list.append(
                        atom_layer_pos-
                        previous_atom_layer)

    output_data_list = np.array([x_pos_list, z_pos_list]).swapaxes(0,1)
    return(output_data_list)

def find_average_distance_between_atoms(
        input_data_list, crop_start=3, crop_end=3):
    data_list = input_data_list[:,0]
    data_list.sort()
    atom_distance_list = data_list[1:]-data_list[:-1]
    normalized_atom_distance_list = atom_distance_list/atom_distance_list.max()
    first_peak_index = np.argmax(
            normalized_atom_distance_list[crop_start:-crop_end] > 0.4) + crop_start
    first_peak = atom_distance_list[first_peak_index]
    return(first_peak)

def combine_clustered_positions_into_layers(
        data_list, layer_distance, combine_layers=True):
    layer_list = []
    one_layer_list = [data_list[0].tolist()]
    for atom_pos in data_list[1:]:
        if np.abs(atom_pos[0] - one_layer_list[-1][0]) < layer_distance:
            one_layer_list.append(atom_pos.tolist())
        else:
            if not (len(one_layer_list) == 1):
                if combine_layers == True:
                    one_layer_list = np.array(one_layer_list).mean(0).tolist()
                layer_list.append(one_layer_list)
            one_layer_list = [atom_pos.tolist()]
    return(layer_list)

def combine_clusters_using_average_distance(data_list, margin=0.5):
    first_peak = find_average_distance_between_atoms(data_list)*margin
    layer_list = combine_clustered_positions_into_layers(
            data_list, first_peak)
    return(layer_list)

def get_peak2d_skimage(image, separation):
    arr_shape = (image.axes_manager._navigation_shape_in_array
            if image.axes_manager.navigation_size > 0
            else [1, ])
    peaks = np.zeros(arr_shape, dtype=object)
    for z, indices in zip(
            image._iterate_signal(),
            image.axes_manager._array_indices_generator()):
        peaks[indices] = peak_local_max(
                z, 
                min_distance=separation)
    return peaks

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def calculate_angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def _get_interpolated2d_from_unregular_data(
        data, new_x_lim=None, new_y_lim=None, upscale=4):
    data = np.array(data)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    if new_x_lim == None:
        new_x_lim = (x.min(), x.max())
    if new_y_lim == None:
        new_y_lim = (y.min(), y.max())
    x_points = (new_x_lim[1]-new_x_lim[0])*upscale
    y_points = (new_y_lim[1]-new_y_lim[0])*upscale
    new_x, new_y = np.mgrid[
            new_x_lim[0]:new_x_lim[1]:x_points*1j,
            new_y_lim[0]:new_y_lim[1]:y_points*1j].astype('float32')
    new_z = interpolate.griddata(
            data[:,0:2], 
            z, 
            (new_x, new_y), 
            method='cubic', 
            fill_value=0.0).astype('float32')
    return(new_x, new_y, new_z)

def plot_zone_vector_and_atom_distance_map(
        image_data,
        distance_data, 
        atom_rows=None,
        distance_data_scale=1,
        atom_list=None,
        extra_marker_list=None,
        clim=None, 
        atom_row_marker=None,
        plot_title='',
        vector_to_plot=None,
        figname="map_data.jpg"):  
    """
    atom_list : list of Atom_Position instances
    extra_marker_list : two arrays of x and y [[x_values], [y_values]]
    """
    fig = plt.figure(figsize=(10,20))
    gs = GridSpec(95,95)

    image_ax = fig.add_subplot(gs[0:45,:])
    distance_ax = fig.add_subplot(gs[45:90,:])
    colorbar_ax = fig.add_subplot(gs[90:,:])
    
    image_clim = _get_clim_from_data(image_data, sigma=2)
    image_cax = image_ax.imshow(image_data)
    image_cax.set_clim(image_clim[0], image_clim[1])
    if atom_rows:
        for atom_row_index, atom_row in enumerate(atom_rows):
            x_pos = atom_row.get_x_position_list()
            y_pos = atom_row.get_y_position_list()
            image_ax.plot(x_pos, y_pos, lw=3, color='blue')
            image_ax.text(
                    atom_row.start_atom.pixel_x, 
                    atom_row.start_atom.pixel_y,
                    str(atom_row_index),
                    color='red')
    image_ax.set_ylim(0, image_data.shape[0])
    image_ax.set_xlim(0, image_data.shape[1])
    image_ax.set_title(plot_title)

    if atom_row_marker:
        atom_row_x = atom_row_marker.get_x_position_list()
        atom_row_y = atom_row_marker.get_y_position_list()
        image_ax.plot(atom_row_x, atom_row_y, color='red', lw=2)

    _make_subplot_map_from_unregular_grid(
        distance_ax,
        distance_data, 
        distance_data_scale=distance_data_scale,
        clim=clim, 
        atom_list=atom_list,
        atom_row_marker=atom_row_marker,
        extra_marker_list=extra_marker_list,
        vector_to_plot=vector_to_plot)
    distance_cax = distance_ax.images[0]

    fig.tight_layout()
    fig.colorbar(distance_cax, cax=colorbar_ax, orientation='horizontal')
    fig.savefig(figname)
    plt.close(fig)

def plot_image_map_line_profile_using_interface_row(
        image_data,
        heatmap_data_list,
        line_profile_data_list,
        interface_row,
        atom_row_list=None,
        data_scale=1,
        atom_list=None,
        extra_marker_list=None,
        clim=None, 
        plot_title='',
        vector_to_plot=None,
        rotate_atom_row_list_90_degrees=False,
        line_profile_prune_outer_values=False,
        figname="map_data.jpg"):  
    """
    atom_list : list of Atom_Position instances
    extra_marker_list : two arrays of x and y [[x_values], [y_values]]
    """
    fig, axarr = plt.subplots(2, 1, figsize=(10,20))
    fig = plt.figure(figsize=(10,20))
    gs = GridSpec(105,95)

    image_ax = fig.add_subplot(gs[0:45,:])
    distance_ax = fig.add_subplot(gs[45:90,:])
    colorbar_ax = fig.add_subplot(gs[90:95,:])
    line_profile_ax = fig.add_subplot(gs[95:,:])
    
    image_y_lim = (0,image_data.shape[0]*data_scale)
    image_x_lim = (0,image_data.shape[1]*data_scale)
        
    image_clim = _get_clim_from_data(image_data, sigma=2)
    image_cax = image_ax.imshow(
            image_data,
            origin='lower',
            extent=[
                image_x_lim[0],
                image_x_lim[1],
                image_y_lim[0],
                image_y_lim[1]])

    image_cax.set_clim(image_clim[0], image_clim[1])
    image_ax.set_xlim(image_x_lim[0], image_x_lim[1])
    image_ax.set_ylim(image_y_lim[0], image_y_lim[1])
    image_ax.set_title(plot_title)

    if not(atom_row_list == None):
        for atom_row in atom_row_list:
            if rotate_atom_row_list_90_degrees:
                atom_row_x = np.array(atom_row.get_x_position_list())
                atom_row_y = np.array(atom_row.get_y_position_list())
                start_x = atom_row_x[0]
                start_y = atom_row_y[0]
                delta_y = (atom_row_x[-1] - atom_row_x[0]) 
                delta_x = -(atom_row_y[-1] - atom_row_y[0]) 
                atom_row_x = np.array([start_x, start_x + delta_x])
                atom_row_y = np.array([start_y, start_y + delta_y])
            else:
                atom_row_x = np.array(atom_row.get_x_position_list())
                atom_row_y = np.array(atom_row.get_y_position_list())
            image_ax.plot(atom_row_x*data_scale, atom_row_y*data_scale, color='red', lw=2)
    
    atom_row_x = np.array(interface_row.get_x_position_list())
    atom_row_y = np.array(interface_row.get_y_position_list())
    image_ax.plot(atom_row_x*data_scale, atom_row_y*data_scale, color='blue', lw=2)

    _make_subplot_map_from_unregular_grid(
        distance_ax,
        heatmap_data_list, 
        distance_data_scale=data_scale,
        clim=clim, 
        atom_list=atom_list,
        atom_row_marker=interface_row,
        extra_marker_list=extra_marker_list,
        vector_to_plot=vector_to_plot)
    distance_cax = distance_ax.images[0]
    distance_ax.plot(atom_row_x*data_scale, atom_row_y*data_scale, color='red', lw=2)

    _make_subplot_line_profile(
        line_profile_ax,
        line_profile_data_list[:,0],
        line_profile_data_list[:,1],
        prune_outer_values=line_profile_prune_outer_values,
        scale=data_scale)

    fig.tight_layout()
    fig.colorbar(distance_cax, cax=colorbar_ax, orientation='horizontal')
    fig.savefig(figname)
    plt.close(fig)

def _make_subplot_line_profile(
        ax,
        x_list,
        y_list,
        scale=1.,
        x_lim=None,
        prune_outer_values=False,
        y_lim=None):
    x_data_list = x_list*scale
    y_data_list = y_list*scale
    if not (prune_outer_values == False):
        x_data_list = x_data_list[prune_outer_values:-prune_outer_values]
        y_data_list = y_data_list[prune_outer_values:-prune_outer_values]
    ax.plot(x_data_list, y_data_list)
    ax.grid()
    if x_lim == None:
        ax.set_xlim(x_data_list.min(), x_data_list.max())
    else:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim == None:
        ax.set_ylim(y_data_list.min(), y_data_list.max())
    else:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.axvline(0, color='red')

def _make_subplot_map_from_unregular_grid(
        ax,
        data, 
        atom_list=None, 
        distance_data_scale=1.,
        clim=None, 
        atom_row_marker=None,
        extra_marker_list=None,
        plot_title='',
        vector_to_plot=None):
    """ Data in the form [(x, y ,z)]"""
    x_lim = (data[0][0][0], data[0][-1][0])
    y_lim = (data[1][0][0], data[1][0][-1])
    cax = ax.imshow(
            data[2].T*distance_data_scale,
            extent=[
                x_lim[0]*distance_data_scale,
                x_lim[1]*distance_data_scale,
                y_lim[0]*distance_data_scale,
                y_lim[1]*distance_data_scale],
            origin='lower',
            cmap='gnuplot2')
    if atom_row_marker:
        atom_row_x = atom_row_marker.get_x_position_list()
        atom_row_y = atom_row_marker.get_y_position_list()
        ax.plot(
                atom_row_x*distance_data_scale, 
                atom_row_y*distance_data_scale, 
                color='red', lw=2)
    if atom_list:
        x = []
        y = []
        for atom in atom_list:
            x.append(atom.pixel_x*distance_data_scale)
            y.append(atom.pixel_y*distance_data_scale)
        ax.scatter(x, y) 
    if extra_marker_list:
        ax.scatter(extra_marker_list[0], extra_marker_list[1], color='red')
    if clim:
        cax.set_clim(clim[0], clim[1])

    if vector_to_plot:
        x0 = x_lim[0] + (x_lim[1] - x_lim[0])/2*0.15
        y0 = y_lim[0] + (y_lim[1] - y_lim[0])/2*0.15
        ax.arrow(
                x0*distance_data_scale, 
                y0*distance_data_scale, 
                vector_to_plot[0]*distance_data_scale, 
                vector_to_plot[1]*distance_data_scale, 
                width=0.20)
    ax.set_xlim(
            data[0][0][0]*distance_data_scale,
            data[0][-1][0]*distance_data_scale)
    ax.set_ylim(
            data[1][0][0]*distance_data_scale,
            data[1][0][-1]*distance_data_scale)

#From Vidars HyperSpy repository
def _line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.
    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of
        the profile is the ceil of the computed length of the scan line.
    Notes
    -----
    This is a utility method meant to be used internally by skimage
    functions. The destination point is included in the profile, in
    contrast to standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = np.ceil(np.hypot(d_row, d_col) + 1)
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    data = np.zeros((2, length, linewidth))
    data[0, :, :] = np.tile(line_col, [linewidth, 1]).T
    data[1, :, :] = np.tile(line_row, [linewidth, 1]).T

    if linewidth != 1:
        # we subtract 1 from linewidth to change from pixel-counting
        # (make this line 3 pixels wide) to point distances (the
        # distance between pixel centers)
        col_width = (linewidth - 1) * np.sin(-theta) / 2
        row_width = (linewidth - 1) * np.cos(theta) / 2
        row_off = np.linspace(-row_width, row_width, linewidth)
        col_off = np.linspace(-col_width, col_width, linewidth)
        data[0, :, :] += np.tile(col_off, [length, 1])
        data[1, :, :] += np.tile(row_off, [length, 1])
    return data

def get_slice_between_two_atoms(image, atom0, atom1, width):
    start_point = atom0.get_pixel_position()
    end_point = atom1.get_pixel_position()
    output_slice = get_arbitrary_slice(image, start_point, end_point, width)
    return(output_slice)

def get_slice_between_four_atoms(image, start_atoms, end_atoms, width):
    start_difference_vector = start_atoms[0].get_pixel_difference(start_atoms[1])
    start_point_x = start_atoms[0].pixel_x - start_difference_vector[0]/2
    start_point_y = start_atoms[0].pixel_y - start_difference_vector[1]/2
    start_point = (start_point_x, start_point_y)

    end_difference_vector = end_atoms[0].get_pixel_difference(end_atoms[1])
    end_point_x = end_atoms[0].pixel_x - end_difference_vector[0]/2
    end_point_y = end_atoms[0].pixel_y - end_difference_vector[1]/2
    end_point = (end_point_x, end_point_y)
    output_slice = get_arbitrary_slice(image, start_point, end_point, width)
    return(output_slice)

def get_arbitrary_slice(image, start_point, end_point, width, debug_figname=None):
    slice_bounds = _line_profile_coordinates(start_point[::-1], end_point[::-1], linewidth=width)

    output_slice = ndimage.map_coordinates(np.transpose(image), slice_bounds)

    if debug_figname:
        fig, axarr = plt.subplots(1,2)
        ax0 = axarr[0]
        ax1 = axarr[1]

        line1_x = [slice_bounds[0][0][0],slice_bounds[0][-1][0]]
        line1_y = [slice_bounds[1][0][0],slice_bounds[1][-1][0]]
        line2_x = [slice_bounds[0][0][-1],slice_bounds[0][-1][-1]]
        line2_y = [slice_bounds[1][0][-1],slice_bounds[1][-1][-1]]

        ax0.imshow(image)
        ax0.plot([start_point[0],end_point[0]],[start_point[1],end_point[1]])
        ax0.plot(line1_x, line1_y)
        ax0.plot(line2_x, line2_y)
        ax1.imshow(np.rot90(np.fliplr(output_slice)))

        ax0.set_ylim(0,image.shape[0])
        ax0.set_xlim(0,image.shape[1])

        ax0.set_title("Original image")
        ax1.set_title("Slice")
        fig.tight_layout()
        fig.savefig("map_coordinates_testing.jpg", dpi=300)
    
    return(output_slice)

def get_point_between_four_atoms(atom_list):
    atom0 = atom_list[0]
    atom1 = atom_list[1]
    atom2 = atom_list[2]
    atom3 = atom_list[3]
    
    x_pos = (atom0.pixel_x + atom1.pixel_x + atom2.pixel_x + atom3.pixel_x)*0.25
    y_pos = (atom0.pixel_y + atom1.pixel_y + atom2.pixel_y + atom3.pixel_y)*0.25
    return((x_pos, y_pos))

def get_point_between_two_atoms(atom_list):
    atom0 = atom_list[0]
    atom1 = atom_list[1]
    
    x_pos = (atom0.pixel_x + atom1.pixel_x)*0.5
    y_pos = (atom0.pixel_y + atom1.pixel_y)*0.5
    return((x_pos, y_pos))

def find_atom_position_between_atom_rows(
        image, 
        atom_row0, 
        atom_row1, 
        orthogonal_zone_vector, 
        integration_width_percent=0.2,
        max_oxygen_sigma_percent=0.2):
    start_atoms_found = False
    start_atom0 = atom_row0.start_atom
    while not start_atoms_found:
        orthogonal_atom0 = start_atom0.get_next_atom_in_zone_vector(orthogonal_zone_vector)
        orthogonal_atom1 = start_atom0.get_previous_atom_in_zone_vector(orthogonal_zone_vector)
        if orthogonal_atom0 in atom_row1.atom_list:
            start_atoms_found = True
            start_atom1 = orthogonal_atom0
        elif orthogonal_atom1 in atom_row1.atom_list:
            start_atoms_found = True
            start_atom1 = orthogonal_atom1
        else:
            start_atom0 = start_atom0.get_next_atom_in_atom_row(atom_row0)
    
    slice_list = []
    
    is_next_atom0 = True
    is_next_atom1 = True

    atom_distance = start_atom0.get_pixel_distance_from_another_atom(start_atom1)
    integration_width = atom_distance*integration_width_percent

    end_atom0 = start_atom0.get_next_atom_in_atom_row(atom_row0)
    end_atom1 = start_atom1.get_next_atom_in_atom_row(atom_row1)
    
    position_x_list = []
    position_y_list = []

    line_segment_list = [] 

    while (end_atom0 and end_atom1):
        output_slice = get_slice_between_four_atoms(
                image, 
                (start_atom0, start_atom1),
                (end_atom0, end_atom1),
                integration_width)

        middle_point = get_point_between_four_atoms(
                [start_atom0, start_atom1, end_atom0, end_atom1])
        position_x_list.append(middle_point[0])
        position_y_list.append(middle_point[1])

        line_segment = (
                get_point_between_two_atoms(
                    [start_atom0, start_atom1]), 
                get_point_between_two_atoms(
                    [end_atom0, end_atom1]))
        line_segment_list.append(line_segment)

        slice_list.append(output_slice)
        start_atom0 = end_atom0
        start_atom1 = end_atom1
        end_atom0 = start_atom0.get_next_atom_in_atom_row(atom_row0)
        end_atom1 = start_atom1.get_next_atom_in_atom_row(atom_row1)

    summed_slices = []
    for slice_data in slice_list:
        summed_slices.append(slice_data.mean(1))
        
    max_oxygen_sigma = max_oxygen_sigma_percent*atom_distance
    centre_value_list = [] 
    for slice_index, summed_slice in enumerate(summed_slices):
        centre_value = _get_centre_value_from_gaussian_model(summed_slice, max_sigma=max_oxygen_sigma, index=slice_index)
        centre_value_list.append(float(centre_value)/len(summed_slice))

    atom_list = []
    for line_segment, centre_value in zip(line_segment_list, centre_value_list):
        end_point = line_segment[1]
        start_point = line_segment[0]

        line_segment_vector = (end_point[0]-start_point[0], end_point[1]-start_point[1])
        atom_vector = (line_segment_vector[0]*centre_value, line_segment_vector[1]*centre_value)
        atom_position = (start_point[0] + atom_vector[0], start_point[1] + atom_vector[1])

        atom = Atom_Position(atom_position[0], atom_position[1])
        
        atom_list.append(atom)

    return(atom_list)

def _get_centre_value_from_gaussian_model(data, max_sigma=None, index=None):
    data = data - data.min()
    data = data/data.max()
    gaussian = hspy.components.Gaussian(
            A=0.5,
            centre=len(data)/2)
    if max_sigma:
        gaussian.sigma.bmax = max_sigma
    signal = hspy.signals.Spectrum(data)
    m = hspy.create_model(signal)
    m.append(gaussian)
    m.fit(fitter='mpfit', bounded=True)
    if False:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(data)
        ax.plot(m.as_signal().data)
        fig.savefig("gri" + str(index) + ".png")
    return(gaussian.centre.value)

def _calculate_distance_between_atoms(atom_list):
    new_x_pos_list, new_y_pos_list, z_pos_list = [], [], []
    for index, atom in enumerate(atom_list):
        if not (index == 0):
            previous_atom = atom_list[index-1]
            previous_x_pos = previous_atom.pixel_x
            previous_y_pos = previous_atom.pixel_y

            x_pos = atom.pixel_x
            y_pos = atom.pixel_y

            new_x_pos = (x_pos + previous_x_pos)*0.5
            new_y_pos = (y_pos + previous_y_pos)*0.5
            z_pos = atom.get_pixel_distance_from_another_atom(previous_atom)

            new_x_pos_list.append(new_x_pos)
            new_y_pos_list.append(new_y_pos)
            z_pos_list.append(z_pos)
    return([new_x_pos_list, new_y_pos_list, z_pos_list])

def _calculate_net_distance_change_between_atoms(atom_list):
    data = _calculate_distance_between_atoms(atom_list)
    x_pos_list = data[0]
    y_pos_list = data[1]
    z_pos_list = data[2]
    new_x_pos_list, new_y_pos_list, new_z_pos_list = [], [], []
    for index, (x_pos,y_pos,z_pos) in enumerate(zip(x_pos_list,y_pos_list,z_pos_list)):
        if not (index == 0):
            previous_x_pos = x_pos_list[index-1]
            previous_y_pos = y_pos_list[index-1]
            previous_z_pos = z_pos_list[index-1]

            new_x_pos = (x_pos + previous_x_pos)*0.5
            new_y_pos = (y_pos + previous_y_pos)*0.5
            new_z_pos = (z_pos - previous_z_pos)

            new_x_pos_list.append(new_x_pos)
            new_y_pos_list.append(new_y_pos)
            new_z_pos_list.append(new_z_pos)
    return([new_x_pos_list, new_y_pos_list, new_z_pos_list])

def _calculate_net_distance_change_between_3d_positions(data_list):
    x_pos_list = data_list[0]
    y_pos_list = data_list[1] 
    z_pos_list = data_list[2]
    new_x_pos_list, new_y_pos_list, new_z_pos_list = [], [], []
    for index, (x_pos, y_pos, z_pos) in enumerate(zip(x_pos_list, y_pos_list, z_pos_list)):
        if not (index == 0):
            previous_x_pos = x_pos_list[index-1]
            previous_y_pos = y_pos_list[index-1]
            previous_z_pos = z_pos_list[index-1]

            new_x_pos = (x_pos + previous_x_pos)*0.5
            new_y_pos = (y_pos + previous_y_pos)*0.5
            new_z_pos = z_pos - previous_z_pos

            new_x_pos_list.append(new_x_pos)
            new_y_pos_list.append(new_y_pos)
            new_z_pos_list.append(new_z_pos)
    return([new_x_pos_list, new_y_pos_list, new_z_pos_list])

def find_atom_positions_for_an_atom_row(
        image, 
        atom_row0,
        atom_row1,
        orthogonal_zone_vector):
    atom_list = find_atom_position_between_atom_rows(
        image, 
        atom_row0, 
        atom_row1, 
        orthogonal_zone_vector)
    position_data = _calculate_net_distance_change_between_atoms(atom_list)
    return(position_data)

def find_atom_positions_for_all_atom_rows(
        image, 
        atom_lattice,
        parallel_zone_vector, 
        orthogonal_zone_vector):
    atom_row_list = atom_lattice.atom_rows_by_zone_vector[parallel_zone_vector]
    x_pos_list, y_pos_list, z_pos_list = [], [], []
    for atom_row_index, atom_row in enumerate(atom_row_list):
        if not (atom_row_index == 0):
            atom_row0 = atom_row_list[atom_row_index-1]
            atom_row1 = atom_row
            position_data = find_atom_positions_for_an_atom_row(
                image, 
                atom_row0,
                atom_row1,
                orthogonal_zone_vector)        
            x_pos_list.extend(position_data[0])
            y_pos_list.extend(position_data[1])
            z_pos_list.extend(position_data[2])
    return([x_pos_list, y_pos_list, z_pos_list])

def _get_clim_from_data(data, sigma=4, ignore_zeros=False, ignore_edges=False):
    if ignore_edges:
        x_lim = int(data.shape[0]*0.05)
        y_lim = int(data.shape[1]*0.05)
        data_array = copy.deepcopy(data[x_lim:-x_lim,y_lim:-y_lim])
    else:
        data_array = copy.deepcopy(data)
    if ignore_zeros:
        data_array = np.ma.masked_values(data_array, 0.0)
    mean = data_array.mean()
    data_variance = data_array.std()*sigma
    clim = (mean-data_variance, mean+data_variance)
    if abs(data_array.min()) < abs(clim[0]):
        clim = list(clim)
        clim[0] = data_array.min()
        clim = tuple(clim)
    if abs(data_array.max()) < abs(clim[1]):
        clim = list(clim)
        clim[1] = data_array.max()
        clim = tuple(clim)
    return(clim)

def find_atom_position_1d_from_distance_list_and_atom_row(
        input_data_list,
        interface_row,
        rebin_data=True):

    x_pos_list = input_data_list[:,0]
    y_pos_list = input_data_list[:,1]
    z_pos_list = input_data_list[:,2]

    x_pos = interface_row.get_x_position_list()
    y_pos = interface_row.get_y_position_list()
    fit = np.polyfit(x_pos, y_pos, 1)
    fit_fn = np.poly1d(fit)
    x_pos_range = max(x_pos) - min(x_pos)
    interface_x = np.linspace(
        min(x_pos)-x_pos_range, 
        max(x_pos)+x_pos_range,
        len(x_pos)*2000)
    interface_y = fit_fn(interface_x)
 
    data_list = []
    for x_pos, y_pos, z_pos in zip(x_pos_list, y_pos_list, z_pos_list):
        closest_distance, direction = interface_row.get_closest_distance_and_angle_to_point(
                (x_pos, y_pos), 
                use_precalculated_line=[interface_x, interface_y],
                plot_debug=False)
        position = closest_distance*math.copysign(1,direction)*-1
        data_list.append([position, z_pos])
    data_list = np.array(data_list) 
    data_list = data_list[data_list[:,0].argsort()] 

    if rebin_data:
        data_list = combine_clusters_using_average_distance(data_list)
    return(data_list)

def _rebin_data_using_histogram_and_peakfinding(x_pos, z_pos):
    peak_position_list = _find_peak_position_using_histogram(
            x_pos, peakgroup=3, amp_thresh=1)
    average_distance = _get_average_distance_between_points(peak_position_list)
    
    x_pos_mask_array = np.ma.array(x_pos)
    z_pos_mask_array = np.ma.array(z_pos)
    new_data_list = []
    for peak_position in peak_position_list:
        mask_data = np.ma.masked_values(
                x_pos, peak_position, atol=average_distance/2)
        x_pos_mask_array.mask = mask_data.mask
        z_pos_mask_array.mask = mask_data.mask
        temp_x_list, temp_z_list = [], []
        for temp_x, temp_z in zip(mask_data.mask*x_pos, mask_data.mask*z_pos):
            if not (temp_x == 0):
                temp_x_list.append(temp_x)
            if not (temp_z == 0):
                temp_z_list.append(temp_z)
        new_data_list.append([
            np.array(temp_x_list).mean(), 
            np.array(temp_z_list).mean()])
    new_data_list = np.array(new_data_list)
    return(new_data_list)

def _find_peak_position_using_histogram(
        data_list, peakgroup=3, amp_thresh=3, debug_plot=False):
    hist = np.histogram(data_list, 1000)
    s = hspy.signals.Signal(hist[0])
    s.axes_manager[-1].scale = hist[1][1] - hist[1][0]
    peak_data = s.find_peaks1D_ohaver(peakgroup=peakgroup, amp_thresh=amp_thresh)
    peak_positions = peak_data[0]['position']+hist[1][0]
    peak_positions.sort()
    if debug_plot:
        fig, ax = plt.subplots()
        ax.plot(s.axes_manager[-1].axis, s.data)
        for peak_position in peak_positions:
            ax.axvline(peak_position)
        fig.savefig(str(np.random.randint(1000,10000)) + ".png")
    return(peak_positions)

def _get_average_distance_between_points(peak_position_list):
    distance_between_peak_list = []
    for peak_index, peak_position in enumerate(peak_position_list):
        if not (peak_index == 0):
            temp_distance = peak_position - peak_position_list[peak_index-1]
            distance_between_peak_list.append(temp_distance)
    average_distance = np.array(distance_between_peak_list).mean()
    return(average_distance)

def plot_stem_image_and_oxygen_position_100_heatmap_for_all_atom_rows(
        image, 
        atom_lattice,
        parallel_zone_vector, 
        orthogonal_zone_vector,
        distance_data_scale=1,
        clim=None,
        plot_title=''):

    plt.ioff()
    data = find_atom_positions_for_all_atom_rows(
        image, 
        atom_lattice,
        parallel_zone_vector, 
        orthogonal_zone_vector)

    atom_position_list = np.array(atom_lattice._get_atom_position_list())
    data[0].extend(atom_position_list[:,0])
    data[1].extend(atom_position_list[:,1])
    data[2].extend(np.zeros(len(atom_position_list[:,0])))
    data = np.swapaxes(data, 0, 1)
    interpolate_x_lim = (0, image.shape[1])
    interpolate_y_lim = (0, image.shape[0])
    new_data = _get_interpolated2d_from_unregular_data(
        data, new_x_lim=interpolate_x_lim, new_y_lim=interpolate_y_lim, upscale=4)
    
    if not clim:
        clim = _get_clim_from_data(data[:,2], sigma=2)
    
    atom_row_list = atom_lattice.atom_rows_by_zone_vector[parallel_zone_vector]
    plot_zone_vector_and_atom_distance_map(
            image,
            new_data,
            atom_rows=[atom_row_list[3]],
            distance_data_scale=distance_data_scale,
            atom_list=atom_lattice.atom_list,
            clim=clim,
            plot_title=plot_title,
            figname=atom_lattice.save_path + "oxygen_position_100.jpg")

def calculate_and_plot_oxygen_100_position_report(
        image,
        atom_lattice,
        parallel_zone_vector, 
        orthogonal_zone_vector,
        atom_row,
        data_scale=None,
        clim=None,
        invert_line_profile=False,
        plot_title=''):

    if data_scale == None:
        data_scale = atom_lattice.pixel_size
    else:
        data_scale = 1.0

    plt.ioff()
    data = find_atom_positions_for_all_atom_rows(
        image, 
        atom_lattice,
        parallel_zone_vector, 
        orthogonal_zone_vector)

    data_for_line_profile = np.swapaxes(np.array(data),0,1)
    line_profile_data_list = find_atom_position_1d_from_distance_list_and_atom_row(
        data_for_line_profile,
        atom_row,
        rebin_data=True)
    line_profile_data_list = np.array(line_profile_data_list)
    if invert_line_profile == True:
        line_profile_data_list[:,0] *= -1
    
    atom_position_list = np.array(atom_lattice._get_atom_position_list())
    data[0].extend(atom_position_list[:,0])
    data[1].extend(atom_position_list[:,1])
    data[2].extend(np.zeros(len(atom_position_list[:,0])))
    data = np.swapaxes(data, 0, 1)
    interpolate_x_lim = (0, image.shape[1])
    interpolate_y_lim = (0, image.shape[0])
    heatmap_data_list = _get_interpolated2d_from_unregular_data(
        data, new_x_lim=interpolate_x_lim, new_y_lim=interpolate_y_lim, upscale=4)

    if not clim:
        clim = _get_clim_from_data(data[:,2], sigma=2)*data_scale
    
    plot_image_map_line_profile_using_interface_row(
        image,
        heatmap_data_list,
        line_profile_data_list,
        atom_row,
        data_scale=data_scale,
        clim=clim,
        plot_title=plot_title,
        figname=atom_lattice.save_path + "oxygen_position_100_line_profile.jpg")
    
    line_profile_dict = {
            'position':(line_profile_data_list[:,0]*data_scale).tolist(),
            'oxygen_position_difference':(line_profile_data_list[:,1]*data_scale).tolist()}
    
    json_filename = atom_lattice.save_path +\
            atom_lattice.path_name +\
            "_oxygen_position_100_line_profile.json"
    with open(json_filename,'w') as fp:
        json.dump(line_profile_dict, fp)

class Atom_Position():
    def __init__(self, x, y):
        self.pixel_x = x
        self.pixel_y = y
        self.nearest_neighbor_list = None
        self.in_atomic_row = []
        self.start_atom = []
        self.end_atom = []
        self.atom_rows = []
        self.tag = ''
        self.old_pixel_x_list = []
        self.old_pixel_y_list = []
        self.sigma_x = 1.0
        self.sigma_y = 1.0
        self.rotation = 0.01

    def get_pixel_position(self):
        return((self.pixel_x, self.pixel_y))

    def get_pixel_difference(self, atom):
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        return((x_distance, y_distance))

    #todo clean up code and split into subfunctions
    def fit_2d_gaussian_with_mask(
            self,
            image_data,
            rotation_enabled=False,
            percent_distance_to_nearest_neighbor=0.40,
            debug_plot=False):
        """ If the Gaussian is centered outside the masked area,
        this function returns False"""
        plt.ioff()
        closest_neighbor = 100000000000000000
        for neighbor_atom in self.nearest_neighbor_list:
            distance = self.get_pixel_distance_from_another_atom(neighbor_atom)
            if distance < closest_neighbor:
                closest_neighbor = distance
        
        delta_d = closest_neighbor*percent_distance_to_nearest_neighbor
        x0 = self.pixel_x - delta_d
        x1 = self.pixel_x + delta_d
        y0 = self.pixel_y - delta_d
        y1 = self.pixel_y + delta_d
        
        if x0 < 0.0:
            x0 = 0
        if y0 < 0.0:
            y0 = 0
        if x1 > image_data.shape[1]:
            x1 = image_data.shape[1]
        if y1 > image_data.shape[0]:
            x1 = image_data.shape[0]
        
        data_slice = copy.deepcopy(image_data[y0:y1,x0:x1])
        data_slice -= data_slice.min()
        data_slice_max = data_slice.max()
        data = data_slice

        mask = _make_circular_mask(
                delta_d, 
                delta_d, 
                data.shape[0],
                data.shape[1], 
                closest_neighbor*percent_distance_to_nearest_neighbor)    
        data = copy.deepcopy(data)
        mask = np.invert(mask)
        data[mask] = 0

        g = hspy.components2d.Gaussian2D(
                centre_x=0.0,
                centre_y=0.0,
                sigma_x=self.sigma_x,
                sigma_y=self.sigma_y,
                rotation=self.rotation,
                A=data_slice_max)


        s = hspy.signals.Image(data)
        s.axes_manager[0].offset = -delta_d
        s.axes_manager[1].offset = -delta_d
        s = hspy.utils.stack([s]*2)
        m = Model2D(s)
        m.append(g)
        m.fit()

        if rotation_enabled:
            g.rotation.free = True
        
        m.fit()

        if debug_plot:
            X,Y = np.meshgrid(
                    np.arange(-delta_d,delta_d,1),
                    np.arange(-delta_d,delta_d,1))
            s_m = g.function(X,Y)

            fig, axarr = plt.subplots(3,2)
            ax0 = axarr[0][0]
            ax1 = axarr[0][1]
            ax2 = axarr[1][0]
            ax3 = axarr[1][1]
            ax4 = axarr[2][0]
            ax5 = axarr[2][1]

            ax0.imshow(data)
            ax1.imshow(s_m)
            ax2.plot(data.sum(0))
            ax2.plot(s_m.sum(0))
            ax3.plot(data.sum(1))
            ax3.plot(s_m.sum(1))

            fig.tight_layout()
            fig.savefig("debug_plot_2d_gaussian_" + str(np.random.randint(1000,10000)) + ".jpg", dpi=400)
            plt.close('all')

        # If the Gaussian centre is located outside the masked region,
        # return False
        dislocation = math.hypot(g.centre_x.value, g.centre_y.value)
        if dislocation > delta_d:
            return(False)
        else:
            g.centre_x.value += self.pixel_x
            g.centre_y.value += self.pixel_y
            return(g)

    def refine_position_using_2d_gaussian(
            self, 
            image_data, 
            rotation_enabled=False,
            percent_distance_to_nearest_neighbor=0.40,
            debug_plot=False):

        for i in range(10):
            g = self.fit_2d_gaussian_with_mask(
                image_data,
                rotation_enabled=rotation_enabled,
                percent_distance_to_nearest_neighbor=percent_distance_to_nearest_neighbor,
                debug_plot=debug_plot)
            if g == False:
                print("Fitting missed")
                if i == 9:
                    #Center of mass calculation
                    new_x, new_y = self.find_center_position_with_center_of_mass_using_mask(
                            image_data,
                            percent_distance_to_nearest_neighbor=\
                                    percent_distance_to_nearest_neighbor)
                    new_sigma_x = self.sigma_x
                    new_sigma_y = self.sigma_y
                    new_rotation = self.rotation
                    break
                else:
                    percent_distance_to_nearest_neighbor *= 0.95
            else:
                new_x = g.centre_x.value
                new_y = g.centre_y.value
                new_rotation = g.rotation.value % 2*math.pi
                new_sigma_x = abs(g.sigma_x.value)
                new_sigma_y = abs(g.sigma_y.value)
                break

        self.old_pixel_x_list.append(self.pixel_x)
        self.old_pixel_y_list.append(self.pixel_y)

        self.pixel_x = new_x
        self.pixel_y = new_y

        self.rotation = new_rotation
        self.sigma_x = new_sigma_x
        self.sigma_y = new_sigma_y
        if self.sigma_x > self.sigma_y:
            self.ellipticity = self.sigma_x/self.sigma_y
        else:
            self.ellipticity = self.sigma_y/self.sigma_x

    def find_center_position_with_center_of_mass_using_mask(
            self,
            image_data,
            percent_distance_to_nearest_neighbor=0.40):
        closest_neighbor = 100000000000000000
        for neighbor_atom in self.nearest_neighbor_list:
            distance = self.get_pixel_distance_from_another_atom(neighbor_atom)
            if distance < closest_neighbor:
                closest_neighbor = distance
        mask = _make_circular_mask(
                self.pixel_y, 
                self.pixel_x, 
                image_data.shape[0],
                image_data.shape[1], 
                closest_neighbor*percent_distance_to_nearest_neighbor)    
        data = copy.deepcopy(image_data)
        mask = np.invert(mask)
        data[mask] = 0

        center_of_mass = self._calculate_center_of_mass(data)

        new_x, new_y = center_of_mass[1], center_of_mass[0]
        return(new_x, new_y)

    def refine_position_using_center_of_mass(
            self, 
            image_data, 
            percent_distance_to_nearest_neighbor=0.40):
        new_x, new_y = self.find_center_position_with_center_of_mass_using_mask(
                image_data,
                percent_distance_to_nearest_neighbor)
        self.old_pixel_x_list.append(self.pixel_x)
        self.old_pixel_y_list.append(self.pixel_y)
        self.pixel_x = new_x
        self.pixel_y = new_y

    def _calculate_center_of_mass(self, data):
        center_of_mass = ndimage.measurements.center_of_mass(data) 
        return(center_of_mass)

    def get_atomic_row_from_zone_vector(self, zone_vector):
        for atomic_row in self.in_atomic_row:
            if atomic_row.zone_vector[0] == zone_vector[0]:
                if atomic_row.zone_vector[1] == zone_vector[1]:
                    return(atomic_row)
        return(False)

    def get_neighbor_atoms_in_atomic_row_from_zone_vector(self, zone_vector):
        atom_row = self.get_atomic_row_from_zone_vector(zone_vector)
        atom_row_atom_neighbor_list = []
        for atom in self.nearest_neighbor_list:
            if atom in atom_row.atom_list:
                atom_row_atom_neighbor_list.append(atom)
        return(atom_row_atom_neighbor_list)

    def is_in_atomic_row(self, zone_direction):
        for atomic_row in self.in_atomic_row:
            if atomic_row.zone_vector[0] == zone_direction[0]:
                if atomic_row.zone_vector[1] == zone_direction[1]:
                    return(True)
        return(False)

    def get_ellipticity_vector(self):
        elli = self.ellipticity - 1
        rot = self.get_rotation_vector()
        vector = (elli*rot[0], elli*rot[1])
        return(vector)

    def get_rotation_vector(self):
        rot = self.rotation
        vector = (
                math.cos(rot),
                math.sin(rot))
        return(vector)

    def get_pixel_distance_from_another_atom(self, atom):
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        total_distance = math.hypot(x_distance, y_distance)
        return(total_distance)

    def pixel_distance_from_point(self, point=(0,0)):
        dist = math.hypot(self.pixel_x - point[0], self.pixel_y - point[1])
        return(dist)

    def get_index_in_atom_row(self, atom_row):
        for atom_index, atom in enumerate(atom_row.atom_list):
            if atom == self:
                return(atom_index)

    def get_next_atom_in_atom_row(self, atom_row):
        current_index = self.get_index_in_atom_row(atom_row)
        if self == atom_row.end_atom:
            return(False)
        else:
            next_atom = atom_row.atom_list[current_index+1]
            return(next_atom)

    def get_previous_atom_in_atom_row(self, atom_row):
        current_index = self.get_index_in_atom_row(atom_row)
        if self == atom_row.start_atom:
            return(False)
        else:
            previous_atom = atom_row.atom_list[current_index-1]
            return(previous_atom)

    def get_next_atom_in_zone_vector(self, zone_vector):
        atom_row = self.get_atomic_row_from_zone_vector(zone_vector)
        if atom_row == False:
            return(False)
        next_atom = self.get_next_atom_in_atom_row(atom_row)
        return(next_atom)

    def get_previous_atom_in_zone_vector(self, zone_vector):
        atom_row = self.get_atomic_row_from_zone_vector(zone_vector)
        if atom_row == False:
            return(False)
        previous_atom = self.get_previous_atom_in_atom_row(atom_row)
        return(previous_atom)

    def can_atom_row_be_reached_through_zone_vector(self, atom_row, zone_vector):
        for test_atom_row in self.atom_rows:
            if test_atom_row.zone_vector == zone_vector:
                for temp_atom in test_atom_row.atom_list:
                    for temp_atom_row in temp_atom.atom_rows:
                        if temp_atom_row == atom_row:
                            return(test_atom_row)
        return(False)

    def get_position_convergence(self, distance_to_first_position=False):
        x_list = self.old_pixel_x_list
        y_list = self.old_pixel_y_list
        distance_list = []
        for index, (x, y) in enumerate(zip(x_list[1:],y_list[1:])):
            if distance_to_first_position:
                previous_x = x_list[0]
                previous_y = y_list[0]
            else:
                previous_x = x_list[index]
                previous_y = y_list[index]
            dist = math.hypot(x - previous_x, y - previous_y)
            distance_list.append(dist)
        return(distance_list)


class Atom_Row():
    def __init__(self, atom_list, zone_vector, atom_lattice):
        self.atom_list = atom_list
        self.zone_vector = zone_vector
        self.atom_lattice = atom_lattice
        self.start_atom = None
        self.end_atom = None
        self._find_start_atom()
        self._find_end_atom()
        self.sort_atoms_by_distance_to_point(self.start_atom.get_pixel_position())

        self.atom_distance_list = self.get_atom_distance_list()
        self._link_atom_to_atom_row()

    def _link_atom_to_atom_row(self):
        for atom in self.atom_list:
            atom.atom_rows.append(self)

    def get_x_position_list(self):
        x_position_list = []
        for atom in self.atom_list:
            x_position_list.append(atom.pixel_x)
        return(x_position_list)

    def get_y_position_list(self):
        y_position_list = []
        for atom in self.atom_list:
            y_position_list.append(atom.pixel_y)
        return(y_position_list)

    def _find_start_atom(self):
        for atom in self.atom_list:
            if self.zone_vector in atom.start_atom:
                self.start_atom = atom
                break

    def _find_end_atom(self):
        for atom in self.atom_list:
            if self.zone_vector in atom.end_atom:
                self.end_atom = atom
                break

    def get_intersecting_atom_from_atom_row(self, atom_row):
        for self_atom in self.atom_list:
            if self_atom in atom_row.atom_list:
                return(self_atom)
        return("Intersecting atom not found")
        
    def sort_atoms_by_distance_to_point(self, point=(0,0)):
        self.atom_list.sort(
                key=operator.methodcaller('pixel_distance_from_point', point))

    def get_slice_between_two_atoms(self, atom1, atom2):
        if not(atom1 in self.atom_list) and not(atom2 in self.atom_list):
            return(False)
        atom1_is_first = None
        for atom in self.atom_list:
            if atom == atom1:
                atom1_is_first = True
                break
            elif atom == atom2:
                atom1_is_first = False
                break
        atom_list = []
        if atom1_is_first:
            while not (atom1 == self.end_atom):
                atom_list.append(atom1)
                atom1 = atom1.get_next_atom_in_atom_row(self)
                if atom1 == atom2:
                    atom_list.append(atom2)
                    break
        return(atom_list)

    def get_atom_distance_list(self):
        atom_distances = []
        for atom_index, atom in enumerate(self.atom_list):
            if not (atom_index == 0):
                distance = atom.get_pixel_distance_from_another_atom(
                        self.atom_list[atom_index-1])
                atom_distances.append(distance)
        return(atom_distances)

    def get_atom_distance_to_next_atom_and_position_list(self):
        """Returns in [(x, y, distance)]"""
        atom_distances = []
        if len(self.atom_list) < 2:
            return(None)
        for atom_index, atom in enumerate(self.atom_list):
            if not (atom_index == 0):
                previous_atom = self.atom_list[atom_index-1]
                difference_vector = previous_atom.get_pixel_difference(atom)
                pixel_x = previous_atom.pixel_x - difference_vector[0]/2
                pixel_y = previous_atom.pixel_y - difference_vector[1]/2
                distance = atom.get_pixel_distance_from_another_atom(
                        previous_atom)
                atom_distances.append([pixel_x, pixel_y, distance])
        atom_distances = np.array(atom_distances)
        return(atom_distances)

    def get_side_edge_atom_rows_between_self_and_another_atom_row(
            self, atom_row, zone_vector):
        start_orthogonal_atom_row = None
        self_atom = self.start_atom
        while start_orthogonal_atom_row == None:
            temp_atom_row = self_atom.can_atom_row_be_reached_through_zone_vector(
                    atom_row, zone_vector)
            if temp_atom_row == False:
                self_atom = self_atom.get_next_atom_in_atom_row(self)
                if self_atom == False:
                    break
            else:
                start_orthogonal_atom_row = temp_atom_row 

        end_orthogonal_atom_row = None
        self_atom = self.end_atom
        while end_orthogonal_atom_row == None:
            temp_atom_row = self_atom.can_atom_row_be_reached_through_zone_vector(
                    atom_row, zone_vector)
            if temp_atom_row == False:
                self_atom = self_atom.get_previous_atom_in_atom_row(self)
                if self_atom == False:
                    break
            else:
                end_orthogonal_atom_row = temp_atom_row 
        return(start_orthogonal_atom_row, end_orthogonal_atom_row)

    def get_net_distance_change_between_atoms(self):
        """Output [(x,y,z)]"""
        if len(self.atom_list) < 3:
            return(None)
        data = self.get_atom_distance_to_next_atom_and_position_list()
        data = np.array(data)
        x_pos_list = data[:,0]
        y_pos_list = data[:,1]
        z_pos_list = data[:,2]
        new_data_list = []
        for index, (x_pos,y_pos,z_pos) in enumerate(zip(x_pos_list,y_pos_list,z_pos_list)):
            if not (index == 0):
                previous_x_pos = x_pos_list[index-1]
                previous_y_pos = y_pos_list[index-1]
                previous_z_pos = z_pos_list[index-1]

                new_x_pos = (x_pos + previous_x_pos)*0.5
                new_y_pos = (y_pos + previous_y_pos)*0.5
                new_z_pos = (z_pos - previous_z_pos)
                new_data_list.append([new_x_pos, new_y_pos, new_z_pos])
        new_data_list = np.array(new_data_list)
        return(new_data_list)

    def get_atom_index(self, check_atom):
        for atom_index, atom in enumerate(self.atom_list):
            if atom ==  check_atom:
                return(atom_index)

    def get_closest_position_to_point(self, point_position, extend_line=False):
        x_pos = self.get_x_position_list()
        y_pos = self.get_y_position_list()

        if (max(x_pos)-min(x_pos)) > (max(y_pos)-min(y_pos)):
            pos_list0 = copy.deepcopy(x_pos)
            pos_list1 = copy.deepcopy(y_pos)
        else:
            pos_list0 = copy.deepcopy(y_pos)
            pos_list1 = copy.deepcopy(x_pos)

        if extend_line:
            reg_results = linregress(pos_list0[:4], pos_list1[:4])
            delta_0 = np.mean((np.array(pos_list0[0:3])-np.array(pos_list0[1:4]).mean()))*40
            delta_1 = reg_results[0]*delta_0
            start_0 = delta_0 + pos_list0[0]
            start_1 = delta_1 + pos_list1[0]
            pos_list0.insert(0, start_0)
            pos_list1.insert(0, start_1)    

            reg_results = linregress(pos_list0[-4:], pos_list1[-4:])
            delta_0 = np.mean((np.array(pos_list0[-3:])-np.array(pos_list0[-4:-1]).mean()))*40
            delta_1 = reg_results[0]*delta_0
            end_0 = delta_0 + pos_list0[-1]
            end_1 = delta_1 + pos_list1[-1]
            pos_list0.append(end_0)
            pos_list1.append(end_1)

        f = interpolate.interp1d(
            pos_list0,
            pos_list1)

#        fig, ax = plt.subplots()
#        ax.scatter(x_pos, y_pos)
#        ax.plot(pos_list0, pos_list1)
#        fig.savefig(str(np.random.randint(1000,20000)) + ".png")
#        plt.close(fig)

        new_pos_list0 = np.linspace(pos_list0[0], pos_list0[-1], len(pos_list0)*100)
        new_pos_list1 = f(new_pos_list0)

        if (max(x_pos)-min(x_pos)) > (max(y_pos)-min(y_pos)):
            new_x = new_pos_list0
            new_y = new_pos_list1
        else:
            new_y = new_pos_list0
            new_x = new_pos_list1

        x_position_point = point_position[0]
        y_position_point = point_position[1]

        dist_x = new_x - x_position_point
        dist_y = new_y - y_position_point

        distance = (dist_x**2 + dist_y**2)**0.5

        closest_index = distance.argmin()        
        closest_point = (new_x[closest_index], new_y[closest_index])
        return(closest_point)

    def get_closest_distance_and_angle_to_point(
            self, 
            point_position, 
            plot_debug=False,
            use_precalculated_line=False):
        x_pos = self.get_x_position_list()
        y_pos = self.get_y_position_list()
        
        if (use_precalculated_line == False):
            fit = np.polyfit(x_pos, y_pos, 1)
            fit_fn = np.poly1d(fit)
            x_pos_range = x_pos[-1] - x_pos[0]
            new_x = np.arange(
                    x_pos[0]-x_pos_range, 
                    x_pos[-1]+x_pos_range,
                    0.00001)
            new_y = fit_fn(new_x)
        else:
            new_x = use_precalculated_line[0]
            new_y = use_precalculated_line[1]

        x_position_point = point_position[0]
        y_position_point = point_position[1]

        dist_x = new_x - x_position_point
        dist_y = new_y - y_position_point

        distance = (dist_x**2 + dist_y**2)**0.5
        closest_index = distance.argmin()
        closest_distance = distance[closest_index]

        point0 = (new_x[closest_index], new_y[closest_index])
        if closest_index == (len(new_x) - 1):
            point1 = (-1*new_x[closest_index-1], -1*new_y[closest_index-1])
        else:
            point1 = (new_x[closest_index+1], new_y[closest_index+1])

        vector0 = (point1[0]-point0[0], point1[1]-point0[1])
        vector1 = (point_position[0]-point0[0], point_position[1]-point0[1])

        direction = np.cross(vector0, vector1)
    
        if plot_debug:
            plt.ioff()
            fig, ax = plt.subplots()
            ax.plot(self.get_x_position_list(), self.get_y_position_list())
            ax.plot(new_x, new_y)
            ax.plot([point_position[0], point0[0]], [point_position[1], point0[1]])
            ax.set_xlim(0,1000)
            ax.set_ylim(0,1000)
            ax.text(0.2,0.2, str(closest_distance*math.copysign(1, direction)))
            fig.savefig(str(np.random.randint(1000,20000)) + ".png")
            plt.close()
        
        return(closest_distance, direction)

    def _plot_debug_atom_row(self):
        fig, ax = plt.subplots(figsize=(10,10))
        cax = ax.imshow(self.atom_lattice.adf_image)
        if self.atom_lattice.plot_clim:
            clim = atom_lattice.plot_clim
            cax.set_clim(clim[0], clim[1])
        for atom_index, atom in enumerate(self.atom_list):
            ax.plot(atom.pixel_x, atom.pixel_y, 'o', color='blue')
            ax.text(atom.pixel_x, atom.pixel_y, str(atom_index))
        ax.set_ylim(0, self.atom_lattice.adf_image.shape[0])
        ax.set_xlim(0, self.atom_lattice.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig("debug_plot_atom_row.jpg")

    def plot_atom_distance(self, figname="atom_distance.png"):
        fig, ax = plt.subplots(figsize=(10,10))
    
# Rename to Sub_Lattice
class Atom_Lattice():
    def __init__(self, atom_position_list, adf_image):
        self.atom_list = []
        for atom_position in atom_position_list:
            atom = Atom_Position(atom_position[0], atom_position[1])
            self.atom_list.append(atom)
        self.zones_axis_average_distances = None
        self.atom_row_list = []
        self.adf_image = adf_image
        self.original_adf_image = None
        self.atom_rows_by_zone_vector = {}
        self.plot_clim = None
        self.tag = ''
        self.save_path = "./"
        self.pixel_size = 1.0
        self.plot_color = 'blue'

    def find_nearest_neighbors(self, nearest_neighbors=9, leafsize=100):
        atom_position_list = self._get_atom_position_list()
        nearest_neighbor_data = sp.spatial.cKDTree(
                atom_position_list,
                leafsize=leafsize)
        for atom in self.atom_list:
            nn_data_list = nearest_neighbor_data.query(
                    atom.get_pixel_position(),
                    nearest_neighbors)
            nn_link_list = []
            #Skipping the first element, since it points to the atom itself
            for nn_link in nn_data_list[1][1:]:
                nn_link_list.append(self.atom_list[nn_link])
            atom.nearest_neighbor_list = nn_link_list

    def get_position_and_ellipticity_vector_for_all_atoms(self):
        x_pos_list = []
        y_pos_list = []
        x_rot_list = [] 
        y_rot_list = [] 

        for atom in self.atom_list:
            x_pos_list.append(atom.pixel_x)
            y_pos_list.append(atom.pixel_y)

            # Maa endres til atom.get_ellipticity_vector
            elli_vector = atom.get_ellipticity_vector()
            x_rot_list.append(elli_vector[0])
            y_rot_list.append(elli_vector[1])

        return(x_pos_list, y_pos_list, x_rot_list, y_rot_list)
    

    def get_atom_row_slice_between_two_rows(self, atom_row1, atom_row2, zone_vector):
        atom_row_start_index = None
        atom_row_end_index = None
        for index, temp_atom_row in enumerate(self.atom_rows_by_zone_vector[zone_vector]):
            if temp_atom_row == atom_row1:
                atom_row_start_index = index
            if temp_atom_row == atom_row2:
                atom_row_end_index = index
        if atom_row_start_index > atom_row_end_index:
            temp_index = atom_row_start_index
            atom_row_start_index = atom_row_end_index
            atom_row_end_index = temp_index
        atom_row_slice = self.atom_rows_by_zone_vector[zone_vector][
                atom_row_start_index:atom_row_end_index]
        return(atom_row_slice)

    def get_atom_list_between_four_atom_rows(
            self, par_atom_row1, par_atom_row2, ort_atom_row1, ort_atom_row2):
        ort_atom_row_slice = self.get_atom_row_slice_between_two_rows(
                ort_atom_row1, ort_atom_row2, ort_atom_row1.zone_vector)
        par_atom_row_slice = self.get_atom_row_slice_between_two_rows(
                par_atom_row1, par_atom_row2, par_atom_row1.zone_vector)

        par_atom_list = []
        for atom_row in par_atom_row_slice:
            par_atom_list.extend(atom_row.atom_list)
        ort_atom_list = []
        for temp_atom_row in ort_atom_row_slice:
            temp_atom_list = []
            for atom in temp_atom_row.atom_list:
                if atom in par_atom_list:
                    temp_atom_list.append(atom)
            ort_atom_list.extend(temp_atom_list)
        return(ort_atom_list)

    def plot_distance_map_from_zone_vector(self, zone_vector, atom_row_marker=None, title='', atom_list=None):
        zone_index = 0
        for index, temp_zone_vector in enumerate(self.zones_axis_average_distances):
            if temp_zone_vector == zone_vector:
                zone_index = index
                break
        atom_row_list = self.atom_rows_by_zone_vector[zone_vector]
        atom_distance_list = []
        for atom_row in atom_row_list:
            atom_distance_list.extend(
                    atom_row.get_atom_distance_to_next_atom_and_position_list())
        
        atom_distance_list = np.array(atom_distance_list)

        interpolate_x_lim = (0, self.adf_image.shape[1])
        interpolate_y_lim = (0, self.adf_image.shape[0])

        data_variance = np.var(atom_distance_list[:,2])
        data_mean = np.mean(atom_distance_list[:,2])
        plot_clim = (data_mean-data_variance*3, data_mean+data_variance*3)
        interpolated_data = _get_interpolated2d_from_unregular_data(
                atom_distance_list, 
                new_x_lim = interpolate_x_lim,
                new_y_lim = interpolate_y_lim)
        plot_zone_vector_and_atom_distance_map(
                self.original_adf_image,
                interpolated_data, 
                atom_rows=[atom_row_list[2]],
                clim=plot_clim,
                plot_title=title,
                atom_row_marker=atom_row_marker,
                vector_to_plot = zone_vector,
                figname=self.save_path + self.tag + "_distance_map_zone" + str(zone_index))

    def plot_distance_map_for_all_zone_vectors(
            self,
            atom_row_marker=None, 
            atom_list=None,
            max_number_of_zone_vectors=5):
        for zone_index, zone_vector in enumerate(self.zones_axis_average_distances):
            if zone_index < max_number_of_zone_vectors:
                self.plot_distance_map_from_zone_vector(
                        zone_vector, 
                        atom_row_marker=atom_row_marker, 
                        atom_list=atom_list,
                        title=str(zone_index) + ", " + str(zone_vector))

    def _make_circular_mask(self, centerX, centerY, imageSizeX, imageSizeY, radius):
        y,x = np.ogrid[-centerX:imageSizeX-centerX, -centerY:imageSizeY-centerY]
        mask = x*x + y*y <= radius*radius
        return(mask)
    
    def find_perpendicular_vector(self, v):
        if v[0] == 0 and v[1] == 0:
            raise ValueError('zero vector')
        return np.cross(v, [1, 0]) 

    def _sort_atom_rows_by_zone_vector(self):
        for zone_vector in self.zones_axis_average_distances:
            temp_atom_row_list = []
            for atom_row in self.atom_row_list:
                if atom_row.zone_vector == zone_vector:
                    temp_atom_row_list.append(atom_row)
            self.atom_rows_by_zone_vector[zone_vector] = temp_atom_row_list
            
        for index, (zone_vector, atom_row_list) in enumerate(self.atom_rows_by_zone_vector.iteritems()):
            length = 100000000
            orthogonal_vector = (length*zone_vector[1], -length*zone_vector[0])
        
            closest_atom_list = []
            for atom_row in atom_row_list:
                closest_atom = 10000000000000000000000000
                for atom in atom_row.atom_list:
                    dist = atom.pixel_distance_from_point(orthogonal_vector)
                    if dist < closest_atom:
                        closest_atom = dist
                closest_atom_list.append(closest_atom)
            atom_row_list.sort(key=dict(zip(atom_row_list, closest_atom_list)).get)

    def refine_atom_positions_using_2d_gaussian(
            self, 
            image_data, 
            percent_distance_to_nearest_neighbor=0.40,
            rotation_enabled=True,
            debug_plot=False):
        for atom in self.atom_list:
            atom.refine_position_using_2d_gaussian(
                    image_data, 
                    rotation_enabled=rotation_enabled,
                    percent_distance_to_nearest_neighbor=percent_distance_to_nearest_neighbor,
                    debug_plot=debug_plot)

    def refine_atom_positions_using_center_of_mass(
            self, image_data, percent_distance_to_nearest_neighbor=0.25):
        for atom_index ,atom in enumerate(self.atom_list):
            atom.refine_position_using_center_of_mass(
                image_data,
                percent_distance_to_nearest_neighbor=percent_distance_to_nearest_neighbor)

    def _get_atom_position_list(self):
        temp_list = []
        for atom in self.atom_list:
            temp_list.append(
                    [atom.pixel_x, atom.pixel_y])

        return(temp_list)

    # Currently not in use
    def _make_nearest_neighbor_distance_statistics(self):
        self.nearest_neighbor_distances = []
        for atom in self.atom_list:
            for nearest_neighbor in atom.nearest_neighbor_list:
                self.nearest_neighbor_distances.append(
                        atom.get_pixel_difference(nearest_neighbor))
#        hist, bins = np.histogram(atom_lattice.nearest_neighbor_distances, bins=50)
#        fig, ax = plt.subplots()
#        ax.hist(self.nearest_neighbor_distances, bins=100)
#        fig.savefig("histogram.png")
                
    def _make_nearest_neighbor_direction_distance_statistics(
            self, 
            nearest_neighbor_histogram_max=0.8,
            debug_figname=''):
        x_pos_distances = []
        y_pos_distances = []
        for atom in self.atom_list:
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = atom.get_pixel_difference(neighbor_atom)
                if not ((distance[0] == 0) and (distance[1] == 0)):
                    x_pos_distances.append(distance[0])
                    y_pos_distances.append(distance[1])
        
        bins = (50,50)
        histogram_range = nearest_neighbor_histogram_max/self.pixel_size
        direction_distance_intensity_hist = np.histogram2d(
                x_pos_distances,
                y_pos_distances,
                bins=bins,
                range=[
                    [-histogram_range,histogram_range],
                    [-histogram_range,histogram_range]])

        if not (debug_figname == ''):
            fig, ax = plt.subplots(figsize=(10,10))
            ax.scatter(x_pos_distances, y_pos_distances)
            ax.set_ylim(-histogram_range,histogram_range)
            ax.set_xlim(-histogram_range,histogram_range)
            fig.savefig(self.save_path + debug_figname)

        hist_scale = direction_distance_intensity_hist[1][1]-\
                direction_distance_intensity_hist[1][0]

        s_direction_distance = hspy.signals.Image(
                direction_distance_intensity_hist[0])
        s_direction_distance.axes_manager[0].offset = -bins[0]/2
        s_direction_distance.axes_manager[1].offset = -bins[1]/2
        s_direction_distance.axes_manager[0].scale = hist_scale
        s_direction_distance.axes_manager[1].scale = hist_scale
        clusters = get_peak2d_skimage(
                s_direction_distance, separation=1)[0]

        shifted_clusters = []
        for cluster in clusters:
            temp_cluster = (
                    round((cluster[0]-bins[0]/2)*hist_scale,2), 
                    round((cluster[1]-bins[1]/2)*hist_scale,2))
            shifted_clusters.append(temp_cluster)

        self.shortest_atom_distance = self._find_shortest_vector(
                shifted_clusters)
        shifted_clusters = self._sort_vectors_by_length(shifted_clusters)

        shifted_clusters = self._remove_parallel_vectors(
                shifted_clusters, 
                tolerance=self.shortest_atom_distance/3.)

        hr_histogram = np.histogram2d(
                x_pos_distances,
                y_pos_distances,
                bins=(250,250),
                range=[
                    [-histogram_range,histogram_range],
                    [-histogram_range,histogram_range]])

        new_zone_vector_list = self._refine_zone_vector_positions(
                shifted_clusters,
                hr_histogram,
                distance_percent=0.5)

        self.zones_axis_average_distances = new_zone_vector_list

    def _refine_zone_vector_positions(
            self, 
            zone_vector_list, 
            histogram,
            distance_percent=0.5):
        """ Refine zone vector positions using center of mass """
        scale = histogram[1][1] - histogram[1][0]
        offset = histogram[1][0]
        closest_distance = math.hypot(zone_vector_list[0][0], zone_vector_list[0][1])*distance_percent/scale

        new_zone_vector_list = []
        for zone_vector in zone_vector_list:
            zone_vector_x = (zone_vector[0]-offset)/scale
            zone_vector_y = (zone_vector[1]-offset)/scale
            circular_mask = _make_circular_mask(
                    zone_vector_x,
                    zone_vector_y,
                    histogram[0].shape[0], 
                    histogram[0].shape[1], 
                    closest_distance)
            center_of_mass = ndimage.measurements.center_of_mass(
                    circular_mask*histogram[0]) 

            new_x_pos = round(center_of_mass[0]*scale+offset,2)
            new_y_pos = round(center_of_mass[1]*scale+offset,2)
            new_zone_vector_list.append((new_x_pos, new_y_pos))
        return(new_zone_vector_list)

    def _sort_vectors_by_length(self, old_vector_list):
        vector_list = copy.deepcopy(old_vector_list)
        zone_vector_distance_list = []
        for vector in vector_list:
            distance = math.hypot(vector[0], vector[1])
            zone_vector_distance_list.append(distance)

        vector_list.sort(key=dict(zip(
            vector_list, zone_vector_distance_list)).get)
        return(vector_list)

    def _find_shortest_vector(self, vector_list):
        shortest_atom_distance = 100000000000000000000000000000
        for vector in vector_list:
            distance = math.hypot(vector[0], vector[1])
            if distance < shortest_atom_distance:
                shortest_atom_distance = distance
        return(shortest_atom_distance)

    def _remove_parallel_vectors(self, old_vector_list, tolerance=7):
        vector_list = copy.deepcopy(old_vector_list)
        element_prune_list = []
        for zone_index, zone_vector in enumerate(vector_list):
            opposite_vector = (-1*zone_vector[0], -1*zone_vector[1])
            for temp_index, temp_zone_vector in enumerate(vector_list[zone_index+1:]):
                dist_x = temp_zone_vector[0]-opposite_vector[0]
                dist_y = temp_zone_vector[1]-opposite_vector[1]
                distance = math.hypot(dist_x, dist_y)
                if distance < tolerance:
                    element_prune_list.append(zone_index+temp_index+1)
        element_prune_list = list(set(element_prune_list))
        element_prune_list.sort()
        element_prune_list.reverse()
        for element_prune in element_prune_list:
            del(vector_list[element_prune])
        return(vector_list)

    def _get_atom_row_list_from_zone_vector(self, zone_vector):
        temp_atom_row_list = []
        for atom_row in self.atom_row_list:
            if atom_row.zone_vector == zone_vector:
                temp_atom_row_list.append(atom_row)
        return(temp_atom_row_list)

    def _generate_all_atom_row_list(self):
        for zone_vector in self.zones_axis_average_distances:
            self._find_all_atomic_rows_from_direction(zone_vector)

    def _find_all_atomic_rows_from_direction(self, zone_vector):
        for atom in self.atom_list:
            already_in_atom_row_with_zone_vector = False
            if not atom.is_in_atomic_row(zone_vector):
                atom_row = self._find_atomic_columns_from_atom(atom, zone_vector)
                if not (len(atom_row) == 1):
                    atom_row_instance = Atom_Row(atom_row, zone_vector, self)
                    for atom in atom_row:
                        atom.in_atomic_row.append(atom_row_instance)
                    self.atom_row_list.append(atom_row_instance)

    def _find_atomic_columns_from_atom(
            self, start_atom, zone_vector, atom_range_factor=0.5):
        atom_range = atom_range_factor*self.shortest_atom_distance
        end_of_atom_row = False
        zone_axis_list1 = [start_atom]
#        start_atom.in_atomic_row.append(zone_vector)
        while not end_of_atom_row:
            atom = zone_axis_list1[-1]
            atoms_within_distance = []
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = neighbor_atom.pixel_distance_from_point(
                        point=(
                            atom.pixel_x+zone_vector[0],
                            atom.pixel_y+zone_vector[1]))
                if distance < atom_range:
                    atoms_within_distance.append([distance, neighbor_atom])
            if atoms_within_distance:
                atoms_within_distance.sort()
                zone_axis_list1.append(atoms_within_distance[0][1])
            if zone_axis_list1[-1] is atom:
                end_of_atom_row = True
                atom.end_atom.append(zone_vector)
        
        zone_vector2 = (-1*zone_vector[0], -1*zone_vector[1])
        start_of_atom_row = False
        zone_axis_list2 = [start_atom]
        while not start_of_atom_row:
            atom = zone_axis_list2[-1]
            atoms_within_distance = []
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = neighbor_atom.pixel_distance_from_point(
                        point=(
                            atom.pixel_x+zone_vector2[0],
                            atom.pixel_y+zone_vector2[1]))
                if distance < atom_range:
                    atoms_within_distance.append([distance, neighbor_atom])
            if atoms_within_distance:
                atoms_within_distance.sort()
                zone_axis_list2.append(atoms_within_distance[0][1])
            if zone_axis_list2[-1] is atom:
                start_of_atom_row = True
                atom.start_atom.append(zone_vector)

        if not (len(zone_axis_list2) == 1):
            zone_axis_list1.extend(zone_axis_list2[1:])
        return(zone_axis_list1)
    
    def find_missing_atoms_from_zone_vector(self, zone_vector, new_atom_tag=''):
        atom_row_list = self.atom_rows_by_zone_vector[zone_vector]

        new_atom_list = []
        new_atom_row_list = []
        for atom_row in atom_row_list:
            temp_new_atom_list = []
            for atom_index, atom in enumerate(atom_row.atom_list[1:]):
                previous_atom = atom_row.atom_list[atom_index]
                difference_vector = previous_atom.get_pixel_difference(atom)
                new_atom_x = previous_atom.pixel_x - difference_vector[0]*0.5
                new_atom_y = previous_atom.pixel_y - difference_vector[1]*0.5
                new_atom = Atom_Position(new_atom_x, new_atom_y)
                new_atom.tag = new_atom_tag
                temp_new_atom_list.append(new_atom)
                new_atom_list.append((new_atom_x, new_atom_y))
            new_atom_row_list.append(temp_new_atom_list)
        return(new_atom_list)

    def plot_atom_row_on_stem_data(self, atom_row_list, figname="atom_row_plot.jpg"):
        fig, ax = plt.subplots(figsize=(10,10))
        cax = ax.imshow(self.adf_image)
        if self.plot_clim:
            cax.set_clim(self.plot_clim[0], self.plot_clim[1])
        for atom_row in atom_row_list:
            x_pos = atom_row.get_x_position_list()
            y_pos = atom_row.get_y_position_list()
            ax.plot(x_pos, y_pos, 'o', color='blue')
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(self.save_path + figname)

    def plot_list_of_positions_on_stem_data(self, x_list, y_list, figname="position_plot.jpg"):
        fig, ax = plt.subplots(figsize=(10,10))
        cax = ax.imshow(self.adf_image)
        if self.plot_clim:
            cax.set_clim(self.plot_clim[0], self.plot_clim[1])
        ax.scatter(x_list, y_list, color='blue')
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(self.save_path + figname)

    def plot_atom_list_on_stem_data(self, 
            atom_list=None, 
            image=None,
            plot_atom_numbers=False, 
            fontsize=12,
            figsize=(20,20),
            figdpi=300,
            figname="atom_plot.jpg"):
        if image == None:
            image = self.original_adf_image
        if atom_list == None:
            atom_list = self.atom_list
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(image)
        if self.plot_clim:
            cax.set_clim(self.plot_clim[0], self.plot_clim[1])
        for atom_index, atom in enumerate(atom_list):
            ax.plot(atom.pixel_x, atom.pixel_y, 'o', color='blue')
            if plot_atom_numbers:
                ax.text(
                        atom.pixel_x, 
                        atom.pixel_y, 
                        str(atom_index), 
                        fontsize=fontsize,
                        color='red')
        ax.set_ylim(0, image.shape[0])
        ax.set_xlim(0, image.shape[1])
        fig.tight_layout()
        fig.savefig(self.save_path + figname, dpi=figdpi)

    def plot_distance_difference_map_for_all_zone_vectors(
            self,
            zone_vector_list=None,
            atom_list_as_zero=None,
            figname="distance_difference.jpg"):

        if zone_vector_list == None:
            zone_vector_list = self.zones_axis_average_distances
        for zone_vector in zone_vector_list:
            self.plot_distance_difference_map_for_zone_vector(
                    zone_vector,
                    atom_list_as_zero=atom_list_as_zero,
                    figname=figname)

    def get_distance_difference_data_list_for_zone_vector(
            self,
            zone_vector):
        x_list, y_list, z_list = [], [], []
        for atom_row in self.atom_rows_by_zone_vector[zone_vector]: 
            data = atom_row.get_net_distance_change_between_atoms()
            if not (data==None):
                x_list.extend(data[:,0])
                y_list.extend(data[:,1])
                z_list.extend(data[:,2])
        return(x_list, y_list, z_list)

    def _add_zero_position_to_data_list_from_atom_list(
            self,
            data_list,
            atom_list):
        atom_x_pos_list = []
        atom_y_pos_list = []
        for atom in atom_list:
            atom_x_pos_list.append(atom.pixel_x)
            atom_y_pos_list.append(atom.pixel_y)
        data_list[0].extend(atom_x_pos_list)
        data_list[1].extend(atom_y_pos_list)
        data_list[2].extend(np.zeros(len(atom_x_pos_list)))

    def get_distance_difference_map_for_zone_vector(
            self, 
            zone_vector,
            atom_list_as_zero=None):

        data_list = self.get_distance_difference_data_list_for_zone_vector(
                zone_vector)
        x_list, y_list, z_list = data_list

        if not (atom_list_as_zero == None):
            self._add_zero_position_to_data_list_from_atom_list(
                    [x_list,y_list,z_list],
                    atom_list_as_zero)
 
        middle_atom_row_index = int(len(self.atom_rows_by_zone_vector)/2)
        middle_atom_row = self.atom_rows_by_zone_vector[zone_vector][middle_atom_row_index]

        data_list = np.array([x_list, y_list, z_list]).swapaxes(0,1)
        interpolate_x_lim = (0, self.adf_image.shape[1])
        interpolate_y_lim = (0, self.adf_image.shape[0])
        new_data = _get_interpolated2d_from_unregular_data(
            data_list,
            new_x_lim=interpolate_x_lim, 
            new_y_lim=interpolate_y_lim, 
            upscale=4)

        return(new_data)

    def plot_distance_difference_map_for_zone_vector(
            self, 
            zone_vector,
            atom_list_as_zero=None,
            figname="distance_difference.jpg"):

        plt.ioff()
        data_list = self.get_distance_difference_map_for_zone_vector(
            zone_vector,
            atom_list_as_zero=atom_list_as_zero)

        data_list = np.array(data_list)

        clim = _get_clim_from_data(data_list[2])

        for index, temp_zone_vector in enumerate(self.zones_axis_average_distances):
            if temp_zone_vector == zone_vector:
                zone_index = index

        atom_row = self.atom_rows_by_zone_vector[zone_vector][0]

        plot_zone_vector_and_atom_distance_map(
            self.original_adf_image,
            data_list,
            atom_rows=[atom_row],
            clim=clim,
            plot_title=str(zone_vector) + ' distance difference list',
            figname=self.save_path + self.tag + "_zone" + str(zone_index) + "_" + figname)

    def save_distance_difference_map_for_zone_vector(
            self, 
            zone_vector,
            atom_list_as_zero=None,
            signal_name="distance_difference.hdf5"):
        data_list = self.get_distance_difference_map_for_zone_vector(
            zone_vector,
            atom_list_as_zero=atom_list_as_zero)

        data_list = np.array(data_list)

        for index, temp_zone_vector in enumerate(self.zones_axis_average_distances):
            if temp_zone_vector == zone_vector:
                zone_index = index

        sig_name = self.save_path + self.tag + "_zone" + str(zone_index) + "_" + signal_name
        
        self.save_map_from_datalist(data_list, self.pixel_size, sig_name)

    def plot_ellipticity(self, interface_row=None, clim=None, figname=''):
        x_list, y_list, z_list = [], [], []
        for atom in self.atom_list:
            x_list.append(atom.pixel_x)
            y_list.append(atom.pixel_y)
            z_list.append(atom.ellipticity)

        data_list = np.array([x_list, y_list, z_list]).swapaxes(0,1)

        # Sometimes the 2D-gaussian fitting is very bad, leading to 
        # a very high sigma_x and very low sigma_y.
        data_list[:,2].clip(0,5, out=data_list[:,2])

        interpolate_x_lim = (0, self.adf_image.shape[1])
        interpolate_y_lim = (0, self.adf_image.shape[0])
        new_data = _get_interpolated2d_from_unregular_data(
            data_list,
            new_x_lim=interpolate_x_lim, 
            new_y_lim=interpolate_y_lim, 
            upscale=4)

        if clim == None:
            clim = (0.9, 2.0)
        
        if not (interface_row == None):
            interface_row = [interface_row]

        plot_zone_vector_and_atom_distance_map(
            self.original_adf_image,
            new_data,
            atom_rows=interface_row,
            clim=clim,
            plot_title='ellipticity',
            figname=self.save_path + self.tag + "_ellipticity" + figname)

    def plot_rotation(self, interface_row=None, clim=None, figname=''):
        x_pos_list, y_pos_list, x_rot_list, y_rot_list = [], [], [], []
        for atom in self.atom_list:
            x_pos_list.append(atom.pixel_x)
            y_pos_list.append(atom.pixel_y)
            rot = atom.get_rotation_vector()
            x_rot_list.append(rot[0])
            y_rot_list.append(rot[1])

        image_data = self.original_adf_image

        fig, axarr = plt.subplots(2, 1, figsize=(10,20))

        image_ax = axarr[0]
        rot_ax = axarr[1]

        image_y_lim = (0,image_data.shape[0])
        image_x_lim = (0,image_data.shape[1])
        image_ax.imshow(image_data)
        image_ax.set_ylim(0, image_data.shape[0])
        image_ax.set_xlim(0, image_data.shape[1])

        rot_ax.quiver(
                x_pos_list,
                y_pos_list,
                x_rot_list,
                y_rot_list,
                scale=40.0,
                headwidth=0.0,
                headlength=0.0,
                headaxislength=0.0,
                pivot='middle')
        rot_ax.imshow(image_data, alpha=0.0)
        rot_ax.set_xlim(min(x_pos_list), max(x_pos_list))
        rot_ax.set_ylim(min(y_pos_list), max(y_pos_list))
        figname=self.save_path + self.tag + "_rotation"
        fig.tight_layout()
        fig.savefig(figname + ".png", dpi=300)

    def plot_ellipticity_rotation(self, interface_row=None, clim=None, figname=''):
        x_pos_list, y_pos_list, x_rot_list, y_rot_list = [], [], [], []
        for atom in self.atom_list:
            x_pos_list.append(atom.pixel_x)
            y_pos_list.append(atom.pixel_y)
            rot = atom.get_ellipticity_vector()
            x_rot_list.append(rot[0])
            y_rot_list.append(rot[1])

        image_data = self.original_adf_image

        fig, axarr = plt.subplots(2, 1, figsize=(10,20))

        image_ax = axarr[0]
        rot_ax = axarr[1]

        image_y_lim = (0,image_data.shape[0])
        image_x_lim = (0,image_data.shape[1])
        image_ax.imshow(image_data)
        image_ax.set_ylim(0, image_data.shape[0])
        image_ax.set_xlim(0, image_data.shape[1])

        rot_ax.quiver(
                x_pos_list,
                y_pos_list,
                x_rot_list,
                y_rot_list,
                scale=40.0,
                headwidth=0.0,
                headlength=0.0,
                headaxislength=0.0,
                pivot='middle')
        rot_ax.imshow(image_data, alpha=0.0)
        rot_ax.set_xlim(min(x_pos_list), max(x_pos_list))
        rot_ax.set_ylim(min(y_pos_list), max(y_pos_list))
        figname=self.save_path + self.tag + "_ellipticity_rotation"
        fig.tight_layout()
        fig.savefig(figname + ".png", dpi=300)

    def plot_all_atom_rows(self, fignameprefix="atom_row"):
        for zone_index, zone_vector in enumerate(self.zones_axis_average_distances):
            fig, ax = plt.subplots(figsize=(20,20))
            cax = ax.imshow(self.adf_image)
            if self.plot_clim:
                cax.set_clim(self.plot_clim[0], self.plot_clim[1])
            for atom_row_index, atom_row in enumerate(
                    self.atom_rows_by_zone_vector[zone_vector]):
                x_pos = atom_row.get_x_position_list()
                y_pos = atom_row.get_y_position_list()
                ax.plot(x_pos, y_pos, lw=3, color='blue')
                ax.text(
                        atom_row.start_atom.pixel_x, 
                        atom_row.start_atom.pixel_y,
                        str(atom_row_index),
                        color='red')
            ax.set_ylim(0, self.adf_image.shape[0])
            ax.set_xlim(0, self.adf_image.shape[1])
            ax.set_title(str(zone_index) + " , " + str(zone_vector))
            fig.tight_layout()
            fig.savefig(self.save_path + fignameprefix + str(zone_index) + ".jpg")

    def plot_atom_row_distance_line(self, atom_row_interface, zone_vector):
        atom_distance_xy_list = []
        for atom in atom_row_interface.atom_list:
            for atom_row in atom.atom_rows:
                if atom_row.zone_vector == zone_vector:
                    atom_index = atom_row.get_atom_index(atom)
                    atom_distance_list = atom_row.get_atom_distance_list()
                    atom_x_range = np.arange(len(atom_distance_list)) - atom_index
                    atom_distance_xy_list.append([atom_x_range, atom_distance_list])

        fig, ax = plt.subplots(figsize=(10,10))
        for atom_distance_xy in atom_distance_xy_list:
            ax.plot(atom_distance_xy[0], atom_distance_xy[1])
        fig.tight_layout()
        fig.savefig(self.save_path + "atom_row_distance_list.jpg")

    def save_map_from_datalist(
            self, 
            data_list,
            data_scale,
            atom_row=None,
            dtype='float32',
            signal_name="datalist_map.hdf5"):
        """data_list : numpy array, 4D"""
        im = hspy.signals.Image(data_list[2])
        x_scale = data_list[0][1][0] - data_list[0][0][0]
        y_scale = data_list[1][0][1] - data_list[1][0][0]
        im.axes_manager[0].scale = x_scale*data_scale
        im.axes_manager[1].scale = y_scale*data_scale
        im.change_dtype('float32')
        if not (atom_row == None):
            im.metadata.add_node('marker.atom_row.x')
            im.metadata.add_node('marker.atom_row.y')
            im.metadata.marker.atom_row.x = atom_row.get_x_position_list()
            im.metadata.marker.atom_row.y = atom_row.get_y_position_list()
        im.save(signal_name, overwrite=True)

    def plot_distance_difference_map_and_line_profile_for_all_zone_vectors(
            self, 
            zone_vector_list=None,
            interface_row=None,
            data_scale=None,
            atom_list_as_zero=None,
            invert_line_profile=False,
            figname="distance_difference_and_lineprofile.jpg",
            save_datafiles=False):
        if zone_vector_list == None:
            zone_vector_list = self.zones_axis_average_distances
        for zone_vector in zone_vector_list:
            self.plot_distance_difference_map_and_line_profile_for_zone_vector(
                zone_vector,
                interface_row=interface_row,
                data_scale=data_scale,
                atom_list_as_zero=atom_list_as_zero,
                invert_line_profile=invert_line_profile,
                figname=figname,
                save_datafiles=save_datafiles)

    def plot_distance_difference_map_and_line_profile_for_zone_vector(
            self, 
            zone_vector,
            interface_row=None,
            data_scale=None,
            atom_list_as_zero=None,
            invert_line_profile=False,
            figname="distance_difference_and_lineprofile.jpg",
            save_datafiles=False):

        if data_scale == None:
            data_scale = self.pixel_size
        else:
            data_scale = 1.0

        if interface_row == None:
            interface_row_index = int(len(self.atom_rows_by_zone_vector[zone_vector])/2.)
            interface_row = self.atom_rows_by_zone_vector[zone_vector][interface_row_index]

        atom_rows = [self.atom_rows_by_zone_vector[zone_vector][1]]

        plt.ioff()
        data_list = self.get_distance_difference_data_list_for_zone_vector(
                zone_vector)

        # Get line profile data
        data_list = np.array(data_list)
        data_for_line_profile = np.swapaxes(np.array(data_list),0,1)
        line_profile_data_list = find_atom_position_1d_from_distance_list_and_atom_row(
            data_for_line_profile,
            interface_row,
            rebin_data=True)
        line_profile_data_list = np.array(line_profile_data_list)
        if invert_line_profile == True:
            line_profile_data_list[:,0] *= -1

        data_list = self.get_distance_difference_map_for_zone_vector(
            zone_vector,
            atom_list_as_zero=atom_list_as_zero)

        clim = _get_clim_from_data(data_list[2]*data_scale)

        for index, temp_zone_vector in enumerate(self.zones_axis_average_distances):
            if temp_zone_vector == zone_vector:
                zone_index = index

        plot_image_map_line_profile_using_interface_row(
            self.original_adf_image,
            data_list,
            line_profile_data_list,
            interface_row,
            atom_row_list=atom_rows,
            data_scale=data_scale,
            clim=clim,
            plot_title=str(zone_vector) + ' distance difference list',
            line_profile_prune_outer_values=2,
            figname=self.save_path + self.tag + "_zone" + str(zone_index) + "_" + figname)
        
        if save_datafiles:
            line_profile_dict = {
                    'position':(
                        line_profile_data_list[:,0]*data_scale).tolist(),
                    'oxygen_position_difference':(
                        line_profile_data_list[:,1]*data_scale).tolist()}

            json_filename = self.save_path + self.tag + "_zone" +\
                    str(zone_index) + "_distance_difference_line_profile.json"
            with open(json_filename,'w') as fp:
                json.dump(line_profile_dict, fp)

            sig_name = self.save_path + self.tag + "_zone" +\
                    str(zone_index) + "_distance_difference.hdf5"
            self.save_map_from_datalist(
                data_list,
                data_scale,
                atom_row=interface_row,
                signal_name=sig_name)

    def get_distance_and_position_list_between_atom_rows(
            self,
            atom_row0,
            atom_row1):
        list_x, list_y, list_z = [],[],[]
        for atom in atom_row0.atom_list:
            pos_x, pos_y = atom_row1.get_closest_position_to_point(
                    (atom.pixel_x, atom.pixel_y), extend_line=True)
            distance = atom.pixel_distance_from_point(point=(pos_x, pos_y))
            list_x.append((pos_x + atom.pixel_x)*0.5)
            list_y.append((pos_y + atom.pixel_y)*0.5)
            list_z.append(distance)
        data_list = np.array([list_x,list_y,list_z])
        return(data_list)

    def get_distance_and_position_list_between_atom_rows_for_zone_vector(
            self, 
            zone_vector):
        atom_row_list = self.atom_rows_by_zone_vector[zone_vector]
        data_list = [[],[],[]]
        for index, atom_row in enumerate(atom_row_list[1:]):
            atom_row_previous = atom_row_list[index]
            row_data_list = self.get_distance_and_position_list_between_atom_rows(
                    atom_row_previous, atom_row)
            data_list[0].extend(row_data_list[0].tolist())
            data_list[1].extend(row_data_list[1].tolist())
            data_list[2].extend(row_data_list[2].tolist())
        data_list = np.array(data_list)
        return(data_list)

    def get_distance_map_and_line_profile_between_atom_rows_from_zone_vector(
            self, 
            atom_row,
            zone_vector):
        data_list = self.get_distance_and_position_list_between_atom_rows_for_zone_vector(
                zone_vector)

       # Get line profile data
        data_for_line_profile = np.swapaxes(np.array(data_list),0,1)
        line_profile_data_list = find_atom_position_1d_from_distance_list_and_atom_row(
            data_for_line_profile,
            atom_row,
            rebin_data=True)
        line_profile_data_list = np.array(line_profile_data_list)
 
        data_list = data_list.swapaxes(0,1)
        interpolate_x_lim = (0, self.adf_image.shape[1])
        interpolate_y_lim = (0, self.adf_image.shape[0])
        new_data = _get_interpolated2d_from_unregular_data(
            data_list,
            new_x_lim=interpolate_x_lim, 
            new_y_lim=interpolate_y_lim, 
            upscale=4)
        return(new_data, line_profile_data_list)

    def plot_distance_map_between_atom_rows_from_zone_vector(
            self, 
            zone_vector,
            interface_row=None,
            save_signal=True,
            line_profile_prune_outer_values=False,
            invert_line_profile=False,
            figname="between_atom_rows.jpg"):
        
        data_scale = self.pixel_size    
    
        if interface_row == None:
            middle_atom_row_index = int(len(self.atom_rows_by_zone_vector)/2)
            interface_row = self.atom_rows_by_zone_vector[zone_vector][middle_atom_row_index]

        temp_data = self.get_distance_map_and_line_profile_between_atom_rows_from_zone_vector(
                interface_row,
                zone_vector)

        data_map, line_profile_data_list = temp_data

        if invert_line_profile == True:
            line_profile_data_list[:,0] *= -1

        clim = _get_clim_from_data(
                data_map[2]*data_scale, sigma=2, ignore_zeros=True, ignore_edges=True)
        
        for index, temp_zone_vector in enumerate(self.zones_axis_average_distances):
            if temp_zone_vector == zone_vector:
                zone_index = index
                break

        atom_rows = [self.atom_rows_by_zone_vector[zone_vector][1]]
        plot_image_map_line_profile_using_interface_row(
            self.original_adf_image,
            data_map,
            line_profile_data_list,
            interface_row,
            data_scale=data_scale,
            clim=clim,
            rotate_atom_row_list_90_degrees=True,
            atom_row_list=atom_rows,
            plot_title=str(zone_vector) + ' distance map between rows',
            line_profile_prune_outer_values=line_profile_prune_outer_values,
            figname=self.save_path + self.tag + "_zone" + str(zone_index) + "_" + figname)
        
        if save_signal:
            save_signal_figname = figname[:-4]
            line_profile_dict = {
                    'position':(
                        line_profile_data_list[:,0]*data_scale).tolist(),
                    self.tag + '_position_difference':(
                        line_profile_data_list[:,1]*data_scale).tolist()}

            json_filename = self.save_path + self.tag + "_zone" +\
                    str(zone_index) + "_" + save_signal_figname + "_line_profile.json"
            with open(json_filename,'w') as fp:
                json.dump(line_profile_dict, fp)

            sig_name = self.save_path + self.tag + "_zone" +\
                    str(zone_index) + "_" + save_signal_figname + ".hdf5"
            self.save_map_from_datalist(
                data_map,
                data_scale,
                atom_row=interface_row,
                signal_name=sig_name)

    def plot_distance_map_between_atom_rows_for_all_zone_vectors(
            self, 
            interface_row=None,
            save_signal=False,
            invert_line_profile=False,
            line_profile_prune_outer_values=False,
            figname="between_atom_rows.jpg"):
        plt.ioff()
        for zone_vector in self.zones_axis_average_distances:
            self.plot_distance_map_between_atom_rows_from_zone_vector(
                    zone_vector,
                    interface_row=interface_row,
                    save_signal=save_signal,
                    invert_line_profile=invert_line_profile,
                    line_profile_prune_outer_values=line_profile_prune_outer_values,
                    figname=figname)

    def _plot_debug_start_end_atoms(self):
        for zone_index, zone_vector in enumerate(self.zones_axis_average_distances):
            fig, ax = plt.subplots(figsize=(10,10))
            cax = ax.imshow(self.adf_image)
            if self.plot_clim:
                cax.set_clim(self.plot_clim[0], self.plot_clim[1])
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom.start_atom:
                    ax.plot(atom.pixel_x, atom.pixel_y, 'o', color='blue')
                    ax.text(atom.pixel_x, atom.pixel_y, str(atom_index))
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom.end_atom:
                    ax.plot(atom.pixel_x, atom.pixel_y, 'o', color='green')
                    ax.text(atom.pixel_x, atom.pixel_y, str(atom_index))
            ax.set_ylim(0, self.adf_image.shape[0])
            ax.set_xlim(0, self.adf_image.shape[1])
            fig.tight_layout()
            fig.savefig(self.save_path + "debug_plot_start_end_atoms_zone" + str(zone_index) + ".jpg")

    def _plot_atom_position_convergence(self, figname='atom_position_convergence.jpg'):
        position_absolute_convergence = []
        position_jump_convergence = []
        for atom in self.atom_list:
            dist0 = atom.get_position_convergence(distance_to_first_position=True)
            dist1 = atom.get_position_convergence()
            position_absolute_convergence.append(dist0)
            position_jump_convergence.append(dist1)

        absolute_convergence = np.array(position_absolute_convergence).mean(axis=0)
        relative_convergence = np.array(position_jump_convergence).mean(axis=0)
        
        fig, axarr = plt.subplots(2,1, sharex=True)
        absolute_ax = axarr[0]
        relative_ax = axarr[1]

        absolute_ax.plot(absolute_convergence)
        relative_ax.plot(relative_convergence)

        absolute_ax.set_ylabel("Average distance from start")
        relative_ax.set_ylabel("Average jump pr. iteration")
        relative_ax.set_xlabel("Refinement step")

        fig.tight_layout()
        fig.savefig(self.save_path + self.tag + "_" + figname)


# Rename to Atom_Lattice
class Material_Structure():
    def __init__(self):
        self.atom_lattice_list = []
        self.adf_image = None
        self.inverted_abf_image = None

    def construct_zone_axes_for_atom_lattices(self, atom_lattice_list=None):
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            construct_zone_axes_from_atom_lattice(atom_lattice)

    def plot_all_atom_lattices(self, image=None, markersize=1, figname="all_atom_lattice.jpg"):
        if image == None:
            image = self.adf_image
        fig, ax = plt.subplots(figsize=(10,10))
        cax = ax.imshow(self.adf_image)
        for atom_lattice in self.atom_lattice_list:
            color = atom_lattice.plot_color
            for atom in atom_lattice.atom_list:
                ax.plot(atom.pixel_x, atom.pixel_y, 'o', markersize=markersize, color=color)
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(figname)

# To be removed
#    def plot_all_atom_lattices_abf(self, markersize=1, figname="all_atom_lattice_abf.jpg"):
#        plt.ioff()
#        fig, ax = plt.subplots(figsize=(10,10))
#        cax = ax.imshow(self.abf_image)
#        for atom_lattice in self.atom_lattice_list:
#            color = atom_lattice.plot_color
#            for atom in atom_lattice.atom_list:
#                ax.plot(atom.pixel_x, atom.pixel_y, 'o', markersize=markersize, color=color)
#        ax.set_ylim(0, self.abf_image.shape[0])
#        ax.set_xlim(0, self.abf_image.shape[1])
#        fig.savefig(figname)

    def plot_atom_distance_maps_for_zone_vectors_and_lattices(
            self,
            atom_lattice_list=None,
            interface_row=None,
            max_number_of_zone_vectors=1):
        plt.ioff()
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            atom_lattice.plot_distance_map_for_all_zone_vectors(
                atom_row_marker=interface_row,
                atom_list=atom_lattice.atom_list,
                max_number_of_zone_vectors=max_number_of_zone_vectors)

    def plot_atom_distance_difference_maps_for_zone_vectors_and_lattices(
            self,
            atom_list_as_zero=None,
            atom_lattice_list=None):
        plt.ioff()
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            atom_lattice.plot_distance_difference_map_for_all_zone_vectors(
                    atom_list_as_zero=atom_list_as_zero)

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


def construct_zone_axes_from_atom_lattice(atom_lattice):
    tag = atom_lattice.tag
    atom_lattice.find_nearest_neighbors(nearest_neighbors=15)
    atom_lattice._make_nearest_neighbor_direction_distance_statistics(debug_figname=tag+"_cat_nn.png")
    atom_lattice._generate_all_atom_row_list()
    atom_lattice._sort_atom_rows_by_zone_vector()
    atom_lattice.plot_all_atom_rows(fignameprefix=tag+"_atom_row")

def refine_atom_lattice(
        atom_lattice, 
        refinement_config_list,
        percent_distance_to_nearest_neighbor):
    tag = atom_lattice.tag

    total_number_of_refinements = 0
    for refinement_config in refinement_config_list:
        total_number_of_refinements += refinement_config[1]

    before_image = refinement_config_list[-1][0]
    atom_lattice.plot_atom_list_on_stem_data(
            atom_lattice.atom_list, 
            image=before_image,
            figname=tag+"_atom_refine0.jpg")
    atom_lattice.find_nearest_neighbors()

    current_counts = 1
    for refinement_config in refinement_config_list:
        image = refinement_config[0]
        number_of_refinements = refinement_config[1]
        refinement_type = refinement_config[2]
        for index in range(1,number_of_refinements+1):
            print(str(current_counts) + "/" + str(total_number_of_refinements))
            if refinement_type == 'gaussian':
                atom_lattice.refine_atom_positions_using_2d_gaussian(
                        image,
                        rotation_enabled=True,
                        percent_distance_to_nearest_neighbor=\
                        percent_distance_to_nearest_neighbor)
            elif refinement_type == 'center_of_mass':
                atom_lattice.refine_atom_positions_using_center_of_mass(
                        image, 
                        percent_distance_to_nearest_neighbor=\
                        percent_distance_to_nearest_neighbor)
            current_counts += 1

    atom_lattice.plot_atom_list_on_stem_data(
            atom_lattice.atom_list, 
            image=image,
            figname=tag+"_atom_refine1.jpg")

# DENNE ER UFERDIG
def make_denoised_stem_signal(signal, invert_signal=False):
    signal.change_dtype('float64')
    temp_signal = signal.deepcopy()
    average_background_data = gaussian_filter(temp_signal.data, 30, mode='nearest')
    background_subtracted = signal.deepcopy().data - average_background_data
    signal_denoised = hspy.signals.Signal(background_subtracted-background_subtracted.min())

    signal_denoised.decomposition()
    signal_denoised = signal_denoised.get_decomposition_model(22)
    if not invert_signal:
        signal_denoised_data = 1./signal_denoised.data
        s_abf = 1./s_abf.data
    else:
        signal_den
    signal_denoised = s_abf_modified2/s_abf_modified2.max()
    s_abf_pca = hspy.signals.Image(s_abf_data_normalized)

def do_pca_on_signal(signal, pca_components=22):
    signal.change_dtype('float64')
    temp_signal = hspy.signals.Signal(signal.data)
    temp_signal.decomposition()
    temp_signal = temp_signal.get_decomposition_model(pca_components)
    temp_signal = hspy.signals.Image(temp_signal.data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)

def subtract_average_background(signal, gaussian_blur=30):
    signal.change_dtype('float64')
    temp_signal = signal.deepcopy()
    average_background_data = gaussian_filter(
            temp_signal.data, gaussian_blur, mode='nearest')
    background_subtracted = signal.deepcopy().data - average_background_data
    temp_signal = hspy.signals.Signal(background_subtracted-background_subtracted.min())
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)

def normalize_signal(signal, invert_signal=False):
    temp_signal = signal.deepcopy()
    if invert_signal:
        temp_signal_data = 1./temp_signal.data
    else:
        temp_signal_data = temp_signal.data
    temp_signal_data = temp_signal_data/temp_signal_data.max()
    temp_signal = hspy.signals.Image(temp_signal_data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)

def run_peak_finding_process_for_single_dataset(
        s_adf_filename, 
        s_abf_filename, 
        peak_separation=0.13, # in nanometers
        refinement_interation_config=None,
        invert_abf_signal=True):
    plt.ioff()
    ################
    s_abf = hspy.load(s_abf_filename)
    s_abf.change_dtype('float64')
    s_abf_modified = subtract_average_background(s_abf)
    s_abf_modified = do_pca_on_signal(s_abf_modified)
    s_abf_modified = normalize_signal(
            s_abf_modified, invert_signal=invert_abf_signal)
    if invert_abf_signal:
        s_abf.data = 1./s_abf.data
    
    ################
    s_adf = hspy.load(s_adf_filename)
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

def run_process_for_adf_image(
        s_adf_filename,
        peak_separation=0.13,
        filter_signal=True):
    plt.ioff()
    ################
    s_adf = hspy.load(s_adf_filename)
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
