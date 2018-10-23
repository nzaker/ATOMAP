import numpy as np
import math
import copy
from tqdm import tqdm, trange
from scipy import interpolate
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import hyperspy.api as hs
from hyperspy.signals import Signal1D, Signal2D
from skimage.morphology import watershed

from atomap.atom_finding_refining import (
    _fit_atom_positions_with_gaussian_model)
from sklearn.cluster import DBSCAN
import logging


# From Vidars HyperSpy repository
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
    contrast to standard NumPy indexing.
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
    return(data)


# Remove atom from image using 2d gaussian model
def remove_atoms_from_image_using_2d_gaussian(
        image, sublattice,
        percent_to_nn=0.40,
        show_progressbar=True):
    """
    Parameters
    ----------
    image : NumPy 2D array
    sublattice : Atomap sublattice object
    percent_to_nn : float
        Percent to nearest neighbor. The function will find the closest
        nearest neighbor to the current atom position, and
        this value times percent_to_nn will be the radius of the mask
        centered on the atom position. Value should be somewhere
        between 0.01 (1%) and 1 (100%). Having a too low value might
        lead to bad fitting.
    show_progressbar : bool, default True

    Returns
    -------
    subtracted_image : NumPy 2D array

    Examples
    --------
    >>> atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> sublattice0 = atom_lattice.sublattice_list[0]
    >>> sublattice0.find_nearest_neighbors()
    >>> import atomap.tools as at
    >>> image_subtracted = at.remove_atoms_from_image_using_2d_gaussian(
    ...        image=atom_lattice.image, sublattice=sublattice0,
    ...        show_progressbar=False)
    >>> import hyperspy.api as hs
    >>> s = hs.signals.Signal2D(image_subtracted)
    >>> s.plot()

    Decrease percent_to_nn, to reduce the effect of overlapping atoms.
    For this dataset it won't change much, but might be very useful for
    real datasets.

    >>> image_subtracted = at.remove_atoms_from_image_using_2d_gaussian(
    ...        image=atom_lattice.image, sublattice=sublattice0,
    ...        percent_to_nn=0.2, show_progressbar=False)

    """
    if sublattice.atom_list[0].nearest_neighbor_list is None:
        raise ValueError(
                "The atom_position objects does not seem to have a "
                "populated nearest neighbor list. "
                "Has sublattice.find_nearest_neighbors() been called?")
    model_image = np.zeros(image.shape)
    X, Y = np.meshgrid(np.arange(
        model_image.shape[1]), np.arange(model_image.shape[0]))
    for atom in tqdm(
            sublattice.atom_list, desc='Subtracting atoms',
            disable=not show_progressbar):
        percent_distance = percent_to_nn
        for i in range(10):
            g_list = _fit_atom_positions_with_gaussian_model(
                [atom],
                image,
                rotation_enabled=True,
                percent_to_nn=percent_distance)
            if g_list is False:
                if i == 9:
                    break
                else:
                    percent_distance *= 0.95
            else:
                g = g_list[0]
                model_image += g.function(X, Y)
                break
    subtracted_image = copy.deepcopy(image) - model_image
    return(subtracted_image)


def get_atom_planes_square(
        sublattice, atom_plane1, atom_plane2,
        interface_atom_plane, zone_vector, debug_plot=False):
    """
    Parameters
    ----------
    sublattice : Atomap sublattice object
    atom_plane1, atom_plane2 : Atomap atom_plane object
    """
    ort_atom_plane1, ort_atom_plane2 = atom_plane1.get_connecting_atom_planes(
        atom_plane2, zone_vector)

    if debug_plot:
        sublattice.plot_atom_plane_on_stem_data(
            [atom_plane1, atom_plane2, ort_atom_plane1, ort_atom_plane2],
            figname="atom_plane_square_debug.jpg")

    atom_list = sublattice.get_atom_list_between_four_atom_planes(
        atom_plane1, atom_plane2, ort_atom_plane1, ort_atom_plane2)

    if debug_plot:
        sublattice.plot_atom_list_on_stem_data(
            atom_list,
            figname="atom_plane_square_atom_list_debug.jpg")

    x_pos_list = []
    y_pos_list = []
    z_pos_list = []
    for atom in atom_list:
        x_pos_list.append(atom.pixel_x)
        y_pos_list.append(atom.pixel_y)
        z_pos_list.append(0)

    data_list = np.array(
        [x_pos_list, y_pos_list, z_pos_list]).swapaxes(0, 1)
    atom_layer_list = project_position_property_sum_planes(
        data_list, interface_atom_plane, rebin_data=True)

    atom_layer_list = np.array(atom_layer_list)[:, 0]
    x_pos_list = []
    z_pos_list = []
    for index, atom_layer_pos in enumerate(atom_layer_list):
        if not (index == 0):
            previous_atom_layer = atom_layer_list[index - 1]
            x_pos_list.append(0.5 * (
                atom_layer_pos +
                previous_atom_layer))
            z_pos_list.append(
                atom_layer_pos -
                previous_atom_layer)

    output_data_list = np.array(
        [x_pos_list, z_pos_list]).swapaxes(0, 1)
    return(output_data_list)


def find_average_distance_between_atoms(
        input_data_list, crop_start=3, crop_end=3, threshold=0.4):
    """Return the distance between monolayers.

    Returns the maximal separation between two adjacent points in
    input_data_list[:,0], as a good approximation for monolayer separation.

    Parameters
    ----------
    input_data_list : NumPy array
        An array where the distance from a point to a line is in
        input_data_list[:, 0]
    crop_start, crop_end : int
        Before and after the index given by crop_start and crop_end, the data
        is ignored. By default 3, to ignore outliers.
    threshold : float, default 0.4
        Atoms with a separation of more than the threshold times the largest
        atom separation will be counted as members of different planes.

    Returns
    -------
    first_peak : float
        The monolayer separation.
    monolayer_sep : array
        An array with monolayer separations
    mean_separation : float
        The mean monolayer separation

    """
    data_list = input_data_list[:, 0]
    data_list.sort()
    atom_distance_list = data_list[1:] - data_list[:-1]
    norm_atom_distance_list = atom_distance_list / atom_distance_list.max()
    is_monolayers = norm_atom_distance_list[crop_start:-crop_end] > threshold
    atoms_wo_outliers = atom_distance_list[crop_start:-crop_end]
    monolayer_sep = atoms_wo_outliers[np.argwhere(is_monolayers)]
    mean_separation = monolayer_sep.mean()
    first_peak_index = np.argmax(is_monolayers) + crop_start
    first_peak = atom_distance_list[first_peak_index]
    if abs(mean_separation - first_peak) > 0.10 * first_peak:
        str1 = '\nThe mean monolayer separation and distance to the first \
                \nmonolayer deviate with more than 10 %. Consider if there \
                \nare too many outliers'
        logging.warning(str1)
    return(first_peak, monolayer_sep, mean_separation)


def combine_clustered_positions_into_layers(
        data_list, layer_distance, combine_layers=True):
    """Combine clustered positions into groups.

    Atoms with a similar distance for a line belong to the same plane parallel
    to this line. Atoms in data_list are grouped based on which plane they
    belong to. If there is only one atom in a layer, it will be disregarded as
    it gives a high uncertainty.

    Parameters
    ----------
    data_list : NumPy array
        An array where the distance from a point to a line is in
        input_data_list[:, 0], and the property of the point (atom)
        is in [:,1]
    layer_distance : float
        The half width of a layer, used to determine which layer an atom
        belongs to.
    combine_layers : bool, default True
        If True, the values for distance and property is averaged for each
        layer.

    Returns
    -------
    A list, layer_list. If combine_layers is True, a list of the average
    position and property of the points in the layer. If False, a nested
    list where each element in layer_list contains a list the atoms (position
    and property) in the layer.

    """
    layer_list = []
    one_layer_list = [data_list[0].tolist()]
    i = 0
    for atom_pos in data_list[1:]:
        if np.abs(atom_pos[0] - one_layer_list[-1][0]) < layer_distance:
            one_layer_list.append(atom_pos.tolist())
            i += 1
        else:
            if not (len(one_layer_list) == 1):
                if combine_layers is True:
                    one_layer_list = np.array(
                        one_layer_list).mean(0).tolist()
                layer_list.append(one_layer_list)
            i += 1
            one_layer_list = [atom_pos.tolist()]
    if combine_layers is True:
        if not (len(one_layer_list) == 1):
            one_layer_list = np.array(one_layer_list).mean(0).tolist()
            layer_list.append(one_layer_list)
    return(layer_list)


def combine_clusters_using_average_distance(data_list, margin=0.5):
    first_peak, monolayer_sep, mean_separation = \
        find_average_distance_between_atoms(data_list)
    layer_list = combine_clustered_positions_into_layers(
        data_list, first_peak * margin)
    return(layer_list)


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def calculate_angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def _get_interpolated2d_from_unregular_data(
        data, new_x_lim=None, new_y_lim=None, upscale=4):
    """
    Parameters
    ----------
    data : numpy array
        Data to be interpolated. Needs to be the shape
        (number of atoms, 3). Where the 3 data points are in the order
        (x-position, y-position, variable).
        To generate this from a list of x-position, y-position
        and variable values:
        data_input = np.array([xpos, ypos, var]).swapaxes(0,1)
    """
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    if new_x_lim is None:
        new_x_lim = (x.min(), x.max())
    if new_y_lim is None:
        new_y_lim = (y.min(), y.max())
    x_points = (new_x_lim[1] - new_x_lim[0]) * upscale
    y_points = (new_y_lim[1] - new_y_lim[0]) * upscale
    new_x, new_y = np.mgrid[
        new_x_lim[0]:new_x_lim[1]:x_points * 1j,
        new_y_lim[0]:new_y_lim[1]:y_points * 1j].astype('float32')
    new_z = interpolate.griddata(
        data[:, 0:2],
        z,
        (new_x, new_y),
        method='cubic',
        fill_value=np.NaN).astype('float32')
    return(new_x, new_y, new_z)


def get_slice_between_two_atoms(image, atom0, atom1, width):
    start_point = atom0.get_pixel_position()
    end_point = atom1.get_pixel_position()
    output_slice = get_arbitrary_slice(image, start_point, end_point, width)
    return(output_slice)


def get_slice_between_four_atoms(image, start_atoms, end_atoms, width):
    start_difference_vector = start_atoms[0].get_pixel_difference(
        start_atoms[1])
    start_point_x = start_atoms[0].pixel_x - start_difference_vector[0] / 2
    start_point_y = start_atoms[0].pixel_y - start_difference_vector[1] / 2
    start_point = (start_point_x, start_point_y)

    end_difference_vector = end_atoms[0].get_pixel_difference(
        end_atoms[1])
    end_point_x = end_atoms[0].pixel_x - end_difference_vector[0] / 2
    end_point_y = end_atoms[0].pixel_y - end_difference_vector[1] / 2
    end_point = (end_point_x, end_point_y)
    output_slice = get_arbitrary_slice(
        image, start_point, end_point, width)
    return(output_slice)


def get_arbitrary_slice(
        image,
        start_point,
        end_point,
        width,
        debug_figname=None):
    slice_bounds = _line_profile_coordinates(
        start_point[::-1], end_point[::-1], linewidth=width)

    output_slice = ndimage.map_coordinates(
        np.transpose(image), slice_bounds)

    if debug_figname:
        fig, axarr = plt.subplots(1, 2)
        ax0 = axarr[0]
        ax1 = axarr[1]

        line1_x = [slice_bounds[0][0][0], slice_bounds[0][-1][0]]
        line1_y = [slice_bounds[1][0][0], slice_bounds[1][-1][0]]
        line2_x = [slice_bounds[0][0][-1], slice_bounds[0][-1][-1]]
        line2_y = [slice_bounds[1][0][-1], slice_bounds[1][-1][-1]]

        ax0.imshow(image)
        ax0.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]])
        ax0.plot(line1_x, line1_y)
        ax0.plot(line2_x, line2_y)
        ax1.imshow(np.rot90(np.fliplr(output_slice)))

        ax0.set_ylim(0, image.shape[0])
        ax0.set_xlim(0, image.shape[1])

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

    x_pos = (
        atom0.pixel_x + atom1.pixel_x +
        atom2.pixel_x + atom3.pixel_x) * 0.25
    y_pos = (
        atom0.pixel_y + atom1.pixel_y +
        atom2.pixel_y + atom3.pixel_y) * 0.25
    return((x_pos, y_pos))


def get_point_between_two_atoms(atom_list):
    atom0 = atom_list[0]
    atom1 = atom_list[1]

    x_pos = (atom0.pixel_x + atom1.pixel_x) * 0.5
    y_pos = (atom0.pixel_y + atom1.pixel_y) * 0.5
    return((x_pos, y_pos))


def find_atom_position_between_atom_planes(
        image,
        atom_plane0,
        atom_plane1,
        orthogonal_zone_vector,
        integration_width_percent=0.2,
        max_oxygen_sigma_percent=0.2):
    start_atoms_found = False
    start_atom0 = atom_plane0.start_atom
    while not start_atoms_found:
        orthogonal_atom0 = start_atom0.get_next_atom_in_zone_vector(
            orthogonal_zone_vector)
        orthogonal_atom1 = start_atom0.get_previous_atom_in_zone_vector(
            orthogonal_zone_vector)
        if orthogonal_atom0 in atom_plane1.atom_list:
            start_atoms_found = True
            start_atom1 = orthogonal_atom0
        elif orthogonal_atom1 in atom_plane1.atom_list:
            start_atoms_found = True
            start_atom1 = orthogonal_atom1
        else:
            start_atom0 = start_atom0.get_next_atom_in_atom_plane(
                atom_plane0)

    slice_list = []

    atom_distance = start_atom0.get_pixel_distance_from_another_atom(
        start_atom1)
    integration_width = atom_distance * integration_width_percent

    end_atom0 = start_atom0.get_next_atom_in_atom_plane(atom_plane0)
    end_atom1 = start_atom1.get_next_atom_in_atom_plane(atom_plane1)

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
        end_atom0 = start_atom0.get_next_atom_in_atom_plane(atom_plane0)
        end_atom1 = start_atom1.get_next_atom_in_atom_plane(atom_plane1)

    summed_slices = []
    for slice_data in slice_list:
        summed_slices.append(slice_data.mean(1))

    max_oxygen_sigma = max_oxygen_sigma_percent * atom_distance
    centre_value_list = []
    for slice_index, summed_slice in enumerate(summed_slices):
        centre_value = _get_centre_value_from_gaussian_model(
            summed_slice, max_sigma=max_oxygen_sigma,
            index=slice_index)
        centre_value_list.append(float(centre_value) / len(summed_slice))

    atom_list = []
    for line_segment, centre_value in zip(
            line_segment_list, centre_value_list):
        end_point = line_segment[1]
        start_point = line_segment[0]

        line_segment_vector = (
            end_point[0] - start_point[0],
            end_point[1] - start_point[1])
        atom_vector = (
            line_segment_vector[0] * centre_value,
            line_segment_vector[1] * centre_value)
        atom_position = (
            start_point[0] + atom_vector[0],
            start_point[1] + atom_vector[1])

        from atom_position_class import Atom_Position

        atom = Atom_Position(atom_position[0], atom_position[1])

        atom_list.append(atom)

    return(atom_list)


def _get_centre_value_from_gaussian_model(data, max_sigma=None, index=None):
    data = data - data.min()
    data = data / data.max()
    gaussian = hs.model.components.Gaussian(
        A=0.5,
        centre=len(data) / 2)
    if max_sigma:
        gaussian.sigma.bmax = max_sigma
    signal = hs.signals.Spectrum(data)
    m = signal.create_model()
    m.append(gaussian)
    m.fit(fitter='mpfit', bounded=True)
    if False:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(data)
        ax.plot(m.as_signal().data)
        fig.savefig("gri" + str(index) + ".png")
    return(gaussian.centre.value)


def _calculate_distance_between_atoms(atom_list):
    new_x_pos_list, new_y_pos_list, z_pos_list = [], [], []
    for index, atom in enumerate(atom_list):
        if not (index == 0):
            previous_atom = atom_list[index - 1]
            previous_x_pos = previous_atom.pixel_x
            previous_y_pos = previous_atom.pixel_y

            x_pos = atom.pixel_x
            y_pos = atom.pixel_y

            new_x_pos = (x_pos + previous_x_pos) * 0.5
            new_y_pos = (y_pos + previous_y_pos) * 0.5
            z_pos = atom.get_pixel_distance_from_another_atom(
                previous_atom)

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
    for index, (x_pos, y_pos, z_pos) in enumerate(
            zip(x_pos_list, y_pos_list, z_pos_list)):
        if not (index == 0):
            previous_x_pos = x_pos_list[index - 1]
            previous_y_pos = y_pos_list[index - 1]
            previous_z_pos = z_pos_list[index - 1]

            new_x_pos = (x_pos + previous_x_pos) * 0.5
            new_y_pos = (y_pos + previous_y_pos) * 0.5
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
    for index, (x_pos, y_pos, z_pos) in enumerate(
            zip(x_pos_list, y_pos_list, z_pos_list)):
        if not (index == 0):
            previous_x_pos = x_pos_list[index - 1]
            previous_y_pos = y_pos_list[index - 1]
            previous_z_pos = z_pos_list[index - 1]

            new_x_pos = (x_pos + previous_x_pos) * 0.5
            new_y_pos = (y_pos + previous_y_pos) * 0.5
            new_z_pos = z_pos - previous_z_pos

            new_x_pos_list.append(new_x_pos)
            new_y_pos_list.append(new_y_pos)
            new_z_pos_list.append(new_z_pos)
    return([new_x_pos_list, new_y_pos_list, new_z_pos_list])


def find_atom_positions_for_an_atom_plane(
        image,
        atom_plane0,
        atom_plane1,
        orthogonal_zone_vector):
    atom_list = find_atom_position_between_atom_planes(
        image,
        atom_plane0,
        atom_plane1,
        orthogonal_zone_vector)
    position_data = _calculate_net_distance_change_between_atoms(
        atom_list)
    return(position_data)


def find_atom_positions_for_all_atom_planes(
        image,
        sublattice,
        parallel_zone_vector,
        orthogonal_zone_vector):
    atom_plane_list = sublattice.atom_planes_by_zone_vector[
        parallel_zone_vector]
    x_pos_list, y_pos_list, z_pos_list = [], [], []
    for atom_plane_index, atom_plane in enumerate(atom_plane_list):
        if not (atom_plane_index == 0):
            atom_plane0 = atom_plane_list[atom_plane_index - 1]
            atom_plane1 = atom_plane
            position_data = find_atom_positions_for_an_atom_plane(
                image,
                atom_plane0,
                atom_plane1,
                orthogonal_zone_vector)
            x_pos_list.extend(position_data[0])
            y_pos_list.extend(position_data[1])
            z_pos_list.extend(position_data[2])
    return([x_pos_list, y_pos_list, z_pos_list])


def _get_clim_from_data(
        data,
        sigma=4,
        ignore_zeros=False,
        ignore_edges=False):
    if ignore_edges:
        x_lim = int(data.shape[0] * 0.05)
        y_lim = int(data.shape[1] * 0.05)
        data_array = copy.deepcopy(data[x_lim:-x_lim, y_lim:-y_lim])
    else:
        data_array = copy.deepcopy(data)
    if ignore_zeros:
        data_array = np.ma.masked_values(data_array, 0.0)
    mean = data_array.mean()
    data_variance = data_array.std() * sigma
    clim = (mean - data_variance, mean + data_variance)
    if abs(data_array.min()) < abs(clim[0]):
        clim = list(clim)
        clim[0] = data_array.min()
        clim = tuple(clim)
    if abs(data_array.max()) < abs(clim[1]):
        clim = list(clim)
        clim[1] = data_array.max()
        clim = tuple(clim)
    return(clim)


def project_position_property_sum_planes(
        input_data_list,
        interface_plane,
        rebin_data=True):
    """
    Project 2D positions onto a 1D plane.
    The 2D positions are found as function of distance
    to the interface_plane. If rebin_data is True,
    the function will attempt to sum the positions belonging
    to the same plane.
    In this case, one will get the positions as a function of
    atomic plane from the interface_plane.

    Note, positions will not be projected on to the interface_plane,
    but on a plane perpendicular to the interface_plane.
    The returned data will give the distance from the projected
    positions to the interface_plane.

    Parameters
    ----------
    input_data_list : Numpy array, [Nx3]
        Numpy array with positions and property value.
        Must be in the from [[x,y,z]], so that the
        x-positions are extracted using input_data_list[:,0].
        y-positions using input_data_list[:,1].
        Property value using input_data_list[:,2].
    interface_plane : Atomap atom_plane object
    rebin_data : bool, optional
        If True, will attempt to combine the data points
        which belong to the same atomic plane.
        The points which belong to the same plane will be
        averaged into a single value for each atomic plane.
        This will give the property value as a function distance
        to the interface_plane.

    Returns
    -------
    Data list : NumPy Array
        Array in the form [[position, property]].

    Example
    -------
    >>> from numpy.random import random
    >>> from atomap.sublattice import Sublattice
    >>> pos = [[x, y] for x in range(9) for y in range(9)]
    >>> sublattice = Sublattice(pos, random((9, 9)))
    >>> sublattice.construct_zone_axes()
    >>> x, y = sublattice.x_position, sublattice.y_position
    >>> z = sublattice.ellipticity
    >>> input_data_list = np.array([x, y, z]).swapaxes(0, 1)
    >>> from atomap.tools import project_position_property_sum_planes
    >>> plane = sublattice.atom_plane_list[10]
    >>> data = project_position_property_sum_planes(input_data_list, plane)
    >>> positions = data[:,0]
    >>> property_values = data[:,1]
    >>> cax = plt.plot(positions, property_values)
    """
    x_pos_list = input_data_list[:, 0]
    y_pos_list = input_data_list[:, 1]
    z_pos_list = input_data_list[:, 2]

    dist = interface_plane.get_closest_distance_and_angle_to_point(
        x_pos_list, y_pos_list)

    data_list = np.stack((dist, z_pos_list)).T
    data_list = data_list[data_list[:, 0].argsort()]

    if rebin_data:
        data_list = combine_clusters_using_average_distance(data_list)
    data_list = np.array(data_list)
    return(data_list)


def _rebin_data_using_histogram_and_peakfinding(x_pos, z_pos):
    peak_position_list = _find_peak_position_using_histogram(
        x_pos, peakgroup=3, amp_thresh=1)
    average_distance = _get_average_distance_between_points(
        peak_position_list)

    x_pos_mask_array = np.ma.array(x_pos)
    z_pos_mask_array = np.ma.array(z_pos)
    new_data_list = []
    for peak_position in peak_position_list:
        mask_data = np.ma.masked_values(
            x_pos, peak_position, atol=average_distance / 2)
        x_pos_mask_array.mask = mask_data.mask
        z_pos_mask_array.mask = mask_data.mask
        temp_x_list, temp_z_list = [], []
        for temp_x, temp_z in zip(
                mask_data.mask * x_pos, mask_data.mask * z_pos):
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
        data_list,
        peakgroup=3,
        amp_thresh=3,
        debug_plot=False):
    hist = np.histogram(data_list, 1000)
    s = hs.signals.Signal(hist[0])
    s.axes_manager[-1].scale = hist[1][1] - hist[1][0]
    peak_data = s.find_peaks1D_ohaver(
        peakgroup=peakgroup, amp_thresh=amp_thresh)
    peak_positions = peak_data[0]['position'] + hist[1][0]
    peak_positions.sort()
    if debug_plot:
        fig, ax = plt.subplots()
        ax.plot(s.axes_manager[-1].axis, s.data)
        for peak_position in peak_positions:
            ax.axvline(peak_position)
        fig.savefig(str(np.random.randint(1000, 10000)) + ".png")
    return(peak_positions)


def _get_average_distance_between_points(peak_position_list):
    distance_between_peak_list = []
    for peak_index, peak_position in enumerate(peak_position_list):
        if not (peak_index == 0):
            temp_distance = peak_position - peak_position_list[
                peak_index - 1]
            distance_between_peak_list.append(temp_distance)
    average_distance = np.array(distance_between_peak_list).mean()
    return(average_distance)


def array2signal1d(array, scale=1.0, offset=0.0):
    signal = Signal1D(array)
    signal.axes_manager[-1].scale = scale
    signal.axes_manager[-1].offset = offset
    return signal


def array2signal2d(numpy_array, scale=1.0, units='',
                   offset_x=0, offset_y=0, rotate_flip=False):
    """Make a HyperSpy 2D signal from a 2D NumPy array

    Parameters
    ----------
    numpy_array : array-like
        Must be 2 dimensions
    scale : float
    units : str
    offset_x : float
    offset_y : float
    rotate_flip : bool, optional
        Default False

    Returns
    -------
    signal : HyperSpy Signal2D

    Examples
    --------
    >>> import atomap.tools as at
    >>> numpy_array = np.random.random((50, 70))
    >>> s = at.array2signal2d(numpy_array)
    >>> s.plot()

    All parameters

    >>> s = at.array2signal2d(
    ...     numpy_array, scale=0.5, units='nm',
    ...     offset_x=7.2, offset_y=-4.1, rotate_flip=True)
    >>> s.plot()

    """
    if rotate_flip:
        signal = Signal2D(np.rot90(np.fliplr(numpy_array)))
    else:
        signal = Signal2D(numpy_array)
    signal.axes_manager[-1].scale = scale
    signal.axes_manager[-2].scale = scale
    signal.axes_manager[-1].units = units
    signal.axes_manager[-2].units = units
    signal.axes_manager[-1].offset = offset_y
    signal.axes_manager[-2].offset = offset_x
    return signal


def _get_n_nearest_neighbors(position_list, nearest_neighbors, leafsize=100):
    """
    Parameters
    ----------
    position_list : NumPy array
        In the form [[x0, y0], [x1, y1], ...].
    nearest_neighbors : int
        The number of closest neighbors which will be returned
    """
    # Need to add one, as the position itself is counted as one neighbor
    nearest_neighbors += 1

    nearest_neighbor_data = cKDTree(
        position_list,
        leafsize=leafsize)
    position_neighbor_list = []
    for position in position_list:
        nn_data_list = nearest_neighbor_data.query(
            position,
            nearest_neighbors)
        # Skipping the first element,
        # since it points to the atom itself
        for position_index in nn_data_list[1][1:]:
            delta_position = position_list[position_index] - position
            position_neighbor_list.append(delta_position)
    return(np.array(position_neighbor_list))


class Fingerprinter:
    """
    Produces a distance-fingerprint from an array of neighbor distance vectors.

    To avoid introducing our own interface we're going to use scikit-learn
    Estimator conventions to name the method, which produces our fingerprint,
    'fit' and store our estimations as attributes with a trailing underscore
    in their names.
    http://scikit-learn.org/stable/developers/contributing.html#fitting
    http://scikit-learn.org/stable/developers/contributing.html#estimated-attributes)

    Attributes
    ----------
    fingerprint_ : array, shape = (n_clusters,)
        The main result. The contents of fingerprint_ can be described as
        the relative distances to neighbours in the generalized neighborhood.
    cluster_centers_ : array, shape = (n_clusters, n_dimensions)
        The cluster center coordinates from which the fingerprint was produced.
    cluster_algo_.labels_ : array, shape = (n_points,)
        Integer labels that denote which cluster each point belongs to.
    """

    def __init__(self, cluster_algo=DBSCAN(eps=0.1, min_samples=10)):
        self._cluster_algo = cluster_algo

    def fit(self, X, max_neighbors=150000):
        """Parameters
        ----------
        X : array, shape = (n_points, n_dimensions)
            This array is typically a transpose of a subset of the returned
            value of sublattice.get_nearest_neighbor_directions_all()
        max_neighbors : int, default 150000
            If the length of X is larger than max_neighbors, X will be reduced
            to max_neighbors. The selection is random. This is done to allow
            very large datasets to be processed, since having X too large
            causes the fitting to use too much memory.

        Notes
        -----
        More information about memory use:
        http://scikit-learn.org/stable/modules/clustering.html#dbscan

        """
        X = np.asarray(X)
        if len(X) > max_neighbors:
            random_indicies = np.random.randint(0, len(X), size=max_neighbors)
            X = X[random_indicies, :]
        n_points, n_dimensions = X.shape

        # Normalize scale so that the clustering algorithm can use constant
        # parameters.
        #
        # E.g. the "eps" parameter in DBSCAN can take advantage of the
        # normalized scale. It specifies the proximity (in the same space
        # as X) required to connect adjacent points into a cluster.
        X_std = X.std()
        X = X / X_std
        cl = self._cluster_algo
        cl.fit(X)

        # The result of a clustering algorithm are labels that indicate which
        # cluster each point belongs to.
        #
        # Labels greater or equal to 0 correspond to valid clusters. A label
        # equal to -1 indicate that this point doesn't belong to any cluster.
        labels = cl.labels_

        # Assert statements here are just to help the reader understand the
        # algorithm by keeping track of the shapes of arrays used.
        assert labels.shape == (n_points,)

        # Number of clusters in labels, removing the -1 label if present.
        n_clusters = len(set(labels) - set([-1]))

        # Cluster centers.
        means = np.zeros((n_clusters, n_dimensions))
        for i in range(n_clusters):
            ith_cluster = X[labels == i]
            means[i] = ith_cluster.mean(axis=0)

        assert means.shape == (n_clusters, n_dimensions)

        # Calculate distances to each center, sort in increasing order.
        dist = np.linalg.norm(means, axis=-1)
        dist.sort()

        # Divide distances by the closest one to get rid of any dependence on
        # the image scale, i.e. produce distance ratios that are unitless.
        dist /= dist[0]

        assert dist.shape == (n_clusters,)

        # Store estimated attributes using the scikit-learn convention.
        # See the docstring of this class.
        self.cluster_centers_ = means * X_std
        self.fingerprint_ = dist
        self.cluster_algo_ = cl

        return self


def integrate(s, points_x, points_y, method='Voronoi', max_radius='Auto',
              show_progressbar=True):
    """Given a spectrum image a set of points and a maximum outer radius,
    this function integrates around each point in an image, using either
    Voronoi cell or watershed segmentation methods.

    Parameters
    ----------
    s : HyperSpy signal or array-like object
        Assuming 2D, 3D or 4D dataset where the spatial dimensions are 2D and
        any remaining dimensions are spectral.
    point_x, point_y : list
        Detailed list of the x and y coordinates of each point of
        interest within the image.
    method : string
        'Voronoi' or 'Watershed'
    max_radius : {'Auto'} int
        A maximum outer radius for each Voronoi Cell.
        If a pixel exceeds this radius it will not be included in the cell.
        This allows analysis of a surface and particles.
        If 'max_radius' is left as 'Auto' then it will be set to the largest
        dimension in the image.
    show_progressbar : bool, optional
        Default True

    Returns
    -------
    integrated_intensity : NumPy array
        An array where dimension 0 is the same length as points, and subsequent
        subsequent dimension are energy dimensions.
    intensity_record : HyperSpy signal, same size as s
        Each pixel/voxel in a particular segment or region has the value of the
        integration, value.
    point_record : NumPy array, same size as image
        Image showing where each integration region is, pixels equating to
        point 0 (integrated_intensity[0]) all have value 0, all pixels
        equating to integrated_intensity[1] all have value 1 etc.

    Examples
    --------

    >>> import atomap.api as am
    >>> from atomap.tools import integrate
    >>> import hyperspy.api as hs
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice(
    ...        image_noise=True)
    >>> image = hs.signals.Signal2D(sublattice.image)
    >>> i_points, i_record, p_record = integrate(
    ...        image,
    ...        points_x=sublattice.x_position,
    ...        points_y=sublattice.y_position, method='Voronoi')
    >>> i_record.plot()

    For a 3 dimensional dataset, with artificial EELS data

    >>> s = am.dummy_data.get_eels_spectrum_survey_image()
    >>> s_eels = am.dummy_data.get_eels_spectrum_map()
    >>> peaks = am.get_atom_positions(s, separation=4)
    >>> i_points, i_record, p_record = integrate(
    ...         s_eels, peaks[:, 0], peaks[:, 1], max_radius=3)

    Note
    ----
    Works in principle with 3D and 4D data sets but will quickly hit a
    memory error with large sizes.

    """
    image = s.__array__()
    if len(image.shape) < 2:
        raise ValueError("s must have at least 2 dimensions")
    intensity_record = np.zeros_like(image, dtype=float)
    currentFeature = np.zeros_like(image.T, dtype=float)
    point_record = np.zeros(image.shape[0:2][::-1], dtype=int)
    integrated_intensity = np.zeros_like(sum(sum(currentFeature.T)))
    integrated_intensity = np.dstack(
        integrated_intensity for i in range(len(points_x)))
    integrated_intensity = np.squeeze(integrated_intensity.T)
    points = np.array((points_y, points_x))
    # Setting max_radius to the width of the image, if none is set.
    if method == 'Voronoi':
        if max_radius == 'Auto':
            max_radius = max(point_record.shape)
        elif max_radius <= 0:
            raise ValueError("max_radius must be higher than 0.")
        distance_log = np.zeros_like(points[0])

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                # For every pixel the distance to all points must be
                # calculated.
                distance_log = ((points[0] - float(i))**2 +
                                (points[1] - float(j))**2)**0.5

                # Next for that pixel the minimum distance to and point should
                # be checked and discarded if too large:
                distMin = np.min(distance_log)
                minIndex = np.argmin(distance_log)

                if distMin >= max_radius:
                    point_record[j][i] = 0
                else:
                    point_record[j][i] = minIndex + 1

    elif method == 'Watershed':
        if len(image.shape) > 2:
            raise ValueError(
                "Currently Watershed method is only implemented for 2D data.")
        points_map = _make_mask(point_record, points[0], points[1])
        point_record = watershed(-image, points_map.T)
        point_record = point_record.T

    else:
        raise NotImplementedError(
                "Oops! You have asked for an unimplemented method.")
    point_record -= 1
    for point in trange(points[0].shape[0], desc='Integrating',
                        disable=not show_progressbar):
        currentMask = (point_record == point)
        currentFeature = currentMask * image.T
        integrated_intensity[point] = sum(sum(currentFeature.T)).T
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if currentMask.T[i][j]:
                    intensity_record[i][j] = integrated_intensity[point]

    if hasattr(s, '_deepcopy_with_new_data'):
        s_intensity_record = s._deepcopy_with_new_data(
            intensity_record, copy_variance=True)
    else:
        s_intensity_record = Signal2D(intensity_record)
    return (integrated_intensity, s_intensity_record, point_record.T)


def _make_mask(image, points_x, points_y):
    mask = np.zeros_like(image)
    for i in range(len(points_x)):
        mask[int(points_y[i])][int(points_x[i])] = i + 1
    return mask


def fliplr_points_and_signal(signal, x_array, y_array):
    """Horizontally flip a set of points and a HyperSpy signal.

    For making sure both the image and points are flipped correctly.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    x_array, y_array : array-like

    Returns
    -------
    flipped_signal : HyperSpy signal
    flipped_x_array, flipped_y_array : NumPy arrays

    Examples
    --------
    >>> import atomap.api as am
    >>> import atomap.tools as to
    >>> sublattice = am.dummy_data.get_distorted_cubic_sublattice()
    >>> s = sublattice.get_atom_list_on_image()
    >>> x, y = sublattice.x_position, sublattice.y_position
    >>> s_flip, x_flip, y_flip = to.fliplr_points_and_signal(s, x, y)

    Plotting the data

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> cax = ax.imshow(s_flip.data, origin='lower',
    ...           extent=s_flip.axes_manager.signal_extent)
    >>> cpoints = ax.scatter(x_flip, y_flip)

    """

    s_out = signal.deepcopy()
    s_out.map(np.fliplr, show_progressbar=False)
    x_array, y_array = fliplr_points_around_signal_centre(
        s_out, x_array, y_array)
    return s_out, x_array, y_array


def fliplr_points_around_signal_centre(signal, x_array, y_array):
    """Horizontally flip a set of points around the centre of a signal.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    x_array, y_array : array-like

    Returns
    -------
    flipped_x_array, flipped_y_array : NumPy arrays

    Examples
    --------
    >>> import atomap.api as am
    >>> import atomap.tools as to
    >>> sublattice = am.dummy_data.get_distorted_cubic_sublattice()
    >>> s = sublattice.get_atom_list_on_image()
    >>> x, y = sublattice.x_position, sublattice.y_position
    >>> x_rot, y_rot = to.fliplr_points_around_signal_centre(s, x, y)

    """
    x_array, y_array = np.array(x_array), np.array(y_array)
    middle_x, middle_y = _get_signal_centre(signal)
    x_array -= middle_x
    y_array -= middle_y
    x_array *= -1
    x_array += middle_x
    y_array += middle_y
    return(x_array, y_array)


def rotate_points_and_signal(signal, x_array, y_array, rotation):
    """Rotate a set of points and a HyperSpy signal.

    For making sure both the image and points are rotated correctly.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    x_array, y_array : array-like
    rotation : scalar

    Returns
    -------
    rotated_signal : HyperSpy signal
    rotated_x_array, rotated_y_array : NumPy arrays

    Examples
    --------
    >>> import atomap.api as am
    >>> import atomap.tools as to
    >>> sublattice = am.dummy_data.get_distorted_cubic_sublattice()
    >>> s = sublattice.get_atom_list_on_image()
    >>> x, y = sublattice.x_position, sublattice.y_position
    >>> s_rot, x_rot, y_rot = to.rotate_points_and_signal(s, x, y, 30)

    Plotting the data

    >>> fig, ax = plt.subplots()
    >>> cax = ax.imshow(s_rot.data, origin='lower',
    ...                 extent=s_rot.axes_manager.signal_extent)
    >>> cpoints = ax.scatter(x_rot, y_rot)

    """
    s_out = signal.deepcopy()
    s_out.map(ndimage.rotate, angle=rotation,
              reshape=False, show_progressbar=False)
    x_array, y_array = rotate_points_around_signal_centre(
        s_out, x_array, y_array, rotation)
    return s_out, x_array, y_array


def rotate_points_around_signal_centre(signal, x_array, y_array, rotation):
    """Rotate a set of points around the centre of a signal.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    x_array, y_array : array-like
    rotation : scalar

    Returns
    -------
    rotated_x_array, rotated_y_array : NumPy arrays

    Examples
    --------
    >>> import atomap.api as am
    >>> import atomap.tools as to
    >>> sublattice = am.dummy_data.get_distorted_cubic_sublattice()
    >>> s = sublattice.get_atom_list_on_image()
    >>> x, y = sublattice.x_position, sublattice.y_position
    >>> x_rot, y_rot = to.rotate_points_around_signal_centre(s, x, y, 30)

    """
    x_array, y_array = np.array(x_array), np.array(y_array)
    middle_x, middle_y = _get_signal_centre(signal)
    x_array -= middle_x
    y_array -= middle_y

    rad_rot = -np.radians(rotation)
    rotation_matrix = np.array([
        [np.cos(rad_rot), -np.sin(rad_rot)],
        [np.sin(rad_rot), np.cos(rad_rot)]])
    xy_matrix = np.array((x_array, y_array))
    xy_matrix = np.dot(xy_matrix.T, rotation_matrix.T)
    x_array = xy_matrix[:, 0]
    y_array = xy_matrix[:, 1]

    x_array += middle_x
    y_array += middle_y
    return(x_array, y_array)


def _get_signal_centre(signal):
    """Get the middle of a signal.

    Parameters
    ----------
    signal : HyperSpy 2D signal

    Returns
    -------
    middle_x, middle_y : centre position of the signal

    """
    sa = signal.axes_manager.signal_axes
    a0_middle = (sa[0].high_value + sa[0].low_value) * 0.5
    a1_middle = (sa[1].high_value + sa[1].low_value) * 0.5
    return(a0_middle, a1_middle)


def _draw_cursor(ax, x, y, xd=10, yd=-30):
    """Draw an arrow resembling a mouse pointer.

    Used for making figures in the documentation.
    Uses the matplotlib ax.annotate to draw the arrow.

    Parameters
    ----------
    ax : matplotlib subplot
    x, y : scalar
        Coordinates for the point of the cursor. In data
        coordinates for the ax. This point can be outside
        the ax extent.
    xd, yd : scalar, optional
        Size of the cursor, in figure display coordinates.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> cax = ax.imshow(np.random.random((100, 100)))
    >>> from atomap.tools import _draw_cursor
    >>> _draw_cursor(ax, 20, 50)

    """
    xd, yd = 10, -30
    arrowprops = dict(
            width=2.9, headwidth=10.3, headlength=15.06,
            edgecolor='white', facecolor='black')
    ax.annotate('', xy=(x, y), xytext=(xd, yd),
                xycoords='data', textcoords='offset pixels',
                arrowprops=arrowprops, annotation_clip=False)


def _update_frame(pos, fig):
    """Update an image frame in a matplotlib FuncAnimation function.

    Will simulate a mouse button press, and update a matplotlib
    annotation.

    Parameters
    ----------
    pos : tuple
        (x, y, press_mouse_button). If press_button is True, a mouse click
        will be done at (x, y), and the cursor will be moved there. If False,
        the cursor will just be moved.
    fig : matplotlib figure object

    """
    ax = fig.axes[0]
    if pos[2]:
        x, y = ax.transData.transform((pos[0], pos[1]))
        fig.canvas.button_press_event(x, y, 1)
    text = ax.texts[0]
    text.xy = (pos[0], pos[1])
    fig.canvas.draw()
    fig.canvas.flush_events()


def _generate_frames_position_list(position_list, num=10):
    """
    Parameters
    ----------
    position_list : list
        Needs to have at least two positions, [[x0, y0], [x1, y1]]
    num : scalar
        Number of points between each position. Default 10.

    Returns
    -------
    frames : list
        Length of num * (len(position_list) - 1) + position_list

    Example
    -------
    >>> from atomap.tools import _generate_frames_position_list
    >>> pos_list = [[10, 20], [65, 10], [31, 71]]
    >>> frames = _generate_frames_position_list(pos_list, num=20)

    """
    frames = []
    for i in range(len(position_list) - 1):
        x0, y0 = position_list[i]
        x1, y1 = position_list[i + 1]
        x_list = np.linspace(x0, x1, num=num, endpoint=False)
        y_list = np.linspace(y0, y1, num=num, endpoint=False)
        frames.append([x0, y0, True])
        for x, y in zip(x_list, y_list):
            frames.append([x, y, False])
    x2, y2 = position_list[-1]
    frames.append([x2, y2, True])
    return frames
