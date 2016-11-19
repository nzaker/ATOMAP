import numpy as np
import math
import copy
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
import hyperspy.api as hs


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
    return(data)


# Remove atom from image using 2d gaussian model
def remove_atoms_from_image_using_2d_gaussian(
        image, sublattice,
        percent_to_nn=0.40):
    model_image = np.zeros(image.shape)
    X, Y = np.meshgrid(np.arange(
        model_image.shape[1]), np.arange(model_image.shape[0]))
    for atom in sublattice.atom_list:
        percent_distance = percent_to_nn
        for i in range(10):
            g = atom.fit_2d_gaussian_with_mask(
                    image,
                    rotation_enabled=True,
                    percent_to_nn=percent_distance)
            if g is False:
                if i == 9:
                    break
                else:
                    percent_distance *= 0.95
            else:
                model_image += g.function(X, Y)
                break
    subtracted_image = copy.deepcopy(image) - model_image
    return(subtracted_image)


# Delete this function?
def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    y, x = np.ogrid[-centerX:imageSizeX-centerX, -centerY:imageSizeY-centerY]
    mask = x*x + y*y <= radius*radius
    return(mask)


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
            previous_atom_layer = atom_layer_list[index-1]
            x_pos_list.append(0.5*(
                        atom_layer_pos +
                        previous_atom_layer))
            z_pos_list.append(
                        atom_layer_pos -
                        previous_atom_layer)

    output_data_list = np.array(
            [x_pos_list, z_pos_list]).swapaxes(0, 1)
    return(output_data_list)


def find_average_distance_between_atoms(
        input_data_list, crop_start=3, crop_end=3):
    data_list = input_data_list[:, 0]
    data_list.sort()
    atom_distance_list = data_list[1:]-data_list[:-1]
    normalized_atom_distance_list = atom_distance_list/atom_distance_list.max()
    first_peak_index = np.argmax(
            normalized_atom_distance_list[
                crop_start:-crop_end] > 0.4) + crop_start
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
                if combine_layers is True:
                    one_layer_list = np.array(
                            one_layer_list).mean(0).tolist()
                layer_list.append(one_layer_list)
            one_layer_list = [atom_pos.tolist()]
    return(layer_list)


def combine_clusters_using_average_distance(data_list, margin=0.5):
    first_peak = find_average_distance_between_atoms(data_list)*margin
    layer_list = combine_clustered_positions_into_layers(
            data_list, first_peak)
    return(layer_list)


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


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
    x_points = (new_x_lim[1]-new_x_lim[0])*upscale
    y_points = (new_y_lim[1]-new_y_lim[0])*upscale
    new_x, new_y = np.mgrid[
            new_x_lim[0]:new_x_lim[1]:x_points*1j,
            new_y_lim[0]:new_y_lim[1]:y_points*1j].astype('float32')
    new_z = interpolate.griddata(
            data[:, 0:2],
            z,
            (new_x, new_y),
            method='cubic',
            fill_value=0.0).astype('float32')
    return(new_x, new_y, new_z)


def get_slice_between_two_atoms(image, atom0, atom1, width):
    start_point = atom0.get_pixel_position()
    end_point = atom1.get_pixel_position()
    output_slice = get_arbitrary_slice(image, start_point, end_point, width)
    return(output_slice)


def get_slice_between_four_atoms(image, start_atoms, end_atoms, width):
    start_difference_vector = start_atoms[0].get_pixel_difference(
            start_atoms[1])
    start_point_x = start_atoms[0].pixel_x - start_difference_vector[0]/2
    start_point_y = start_atoms[0].pixel_y - start_difference_vector[1]/2
    start_point = (start_point_x, start_point_y)

    end_difference_vector = end_atoms[0].get_pixel_difference(
            end_atoms[1])
    end_point_x = end_atoms[0].pixel_x - end_difference_vector[0]/2
    end_point_y = end_atoms[0].pixel_y - end_difference_vector[1]/2
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
            atom2.pixel_x + atom3.pixel_x)*0.25
    y_pos = (
            atom0.pixel_y + atom1.pixel_y +
            atom2.pixel_y + atom3.pixel_y)*0.25
    return((x_pos, y_pos))


def get_point_between_two_atoms(atom_list):
    atom0 = atom_list[0]
    atom1 = atom_list[1]

    x_pos = (atom0.pixel_x + atom1.pixel_x)*0.5
    y_pos = (atom0.pixel_y + atom1.pixel_y)*0.5
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
    integration_width = atom_distance*integration_width_percent

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

    max_oxygen_sigma = max_oxygen_sigma_percent*atom_distance
    centre_value_list = []
    for slice_index, summed_slice in enumerate(summed_slices):
        centre_value = _get_centre_value_from_gaussian_model(
                summed_slice, max_sigma=max_oxygen_sigma,
                index=slice_index)
        centre_value_list.append(float(centre_value)/len(summed_slice))

    atom_list = []
    for line_segment, centre_value in zip(
            line_segment_list, centre_value_list):
        end_point = line_segment[1]
        start_point = line_segment[0]

        line_segment_vector = (
                end_point[0]-start_point[0],
                end_point[1]-start_point[1])
        atom_vector = (
                line_segment_vector[0]*centre_value,
                line_segment_vector[1]*centre_value)
        atom_position = (
                start_point[0] + atom_vector[0],
                start_point[1] + atom_vector[1])

        from atom_position_class import Atom_Position

        atom = Atom_Position(atom_position[0], atom_position[1])

        atom_list.append(atom)

    return(atom_list)


def _get_centre_value_from_gaussian_model(data, max_sigma=None, index=None):
    data = data - data.min()
    data = data/data.max()
    gaussian = hs.model.components.Gaussian(
            A=0.5,
            centre=len(data)/2)
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
            previous_atom = atom_list[index-1]
            previous_x_pos = previous_atom.pixel_x
            previous_y_pos = previous_atom.pixel_y

            x_pos = atom.pixel_x
            y_pos = atom.pixel_y

            new_x_pos = (x_pos + previous_x_pos)*0.5
            new_y_pos = (y_pos + previous_y_pos)*0.5
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
    for index, (x_pos, y_pos, z_pos) in enumerate(
            zip(x_pos_list, y_pos_list, z_pos_list)):
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
            atom_plane0 = atom_plane_list[atom_plane_index-1]
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
        x_lim = int(data.shape[0]*0.05)
        y_lim = int(data.shape[1]*0.05)
        data_array = copy.deepcopy(data[x_lim:-x_lim, y_lim:-y_lim])
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
    Numpy array, [[position, property]].

    Example
    -------
    >>>> data = project_position_property_sum_planes(
            input_data,
            interface_plane)
    >>>> positions = data[:,0]
    >>>> property_values = data[:,0]
    >>>> plt.plot(positions, property_values)
    """
    x_pos_list = input_data_list[:, 0]
    y_pos_list = input_data_list[:, 1]
    z_pos_list = input_data_list[:, 2]

    x_pos = interface_plane.get_x_position_list()
    y_pos = interface_plane.get_y_position_list()
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
        closest_distance, direction =\
                interface_plane.get_closest_distance_and_angle_to_point(
                    (x_pos, y_pos),
                    use_precalculated_line=[interface_x, interface_y],
                    plot_debug=False)
        position = closest_distance*math.copysign(1, direction)*-1
        data_list.append([position, z_pos])
    data_list = np.array(data_list)
    data_list = data_list[data_list[:, 0].argsort()]

    if rebin_data:
        data_list = combine_clusters_using_average_distance(data_list)
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
                x_pos, peak_position, atol=average_distance/2)
        x_pos_mask_array.mask = mask_data.mask
        z_pos_mask_array.mask = mask_data.mask
        temp_x_list, temp_z_list = [], []
        for temp_x, temp_z in zip(
                mask_data.mask*x_pos, mask_data.mask*z_pos):
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
                    peak_index-1]
            distance_between_peak_list.append(temp_distance)
    average_distance = np.array(distance_between_peak_list).mean()
    return(average_distance)
