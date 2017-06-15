from scipy.ndimage.filters import gaussian_filter
from hyperspy.signals import Signal2D
import hyperspy.api as hs
import numpy as np
from skimage.feature import peak_local_max
from copy import deepcopy
import math

from atomap.external.gaussian2d import Gaussian2D


def get_atom_positions(
        signal,
        separation=5,
        threshold_rel=0.02,
        pca=False,
        subtract_background=False,
        normalize_intensity=False,
        remove_too_close_atoms=True):
    """
    Find the most intense features in a HyperSpy signal, where the
    features has to be separated by a minimum distance.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    separation : number
        Minimum separation between the features.
    threshold_rel : float, default 0.02
    pca : bool, default False
        Do PCA on the signal before doing the peak finding.
    subtract_background : bool, default False
        Subtract the average background from the signal before
        doing the peak finding.
    normalize_intensity : bool, default False
    remove_too_close_atoms : bool, default True
        Will attempt to find and remove atoms which are too close to
        eachother, i.e. less than separation.

    Returns
    -------
    NumPy array, list of the most intense atom positions.

    Example
    -------
    If s is a single signal
    >>> atom_positions = get_atom_positions(s, 5)
    >>> peak_x = atom_positions[:,0]
    >>> peak_y = atom_positions[:,1]
    """
    if pca:
        signal = do_pca_on_signal(signal)
    if subtract_background:
        signal = subtract_average_background(signal)
    if normalize_intensity:
        signal = normalize_signal(signal)

    image_data = signal.data
    if image_data.dtype is np.dtype('float16'):
        image_data = image_data.astype('float32')
    if image_data.dtype is np.dtype('int8'):
        image_data = image_data.astype('int32')
    if image_data.dtype is np.dtype('int16'):
        image_data = image_data.astype('int32')

    temp_positions = peak_local_max(
            image=image_data,
            min_distance=int(separation),
            threshold_rel=threshold_rel,
            indices=True)

    # The X- and Y-axes are switched in HyperSpy compared to NumPy
    # so we need to flip them here
    atom_positions = np.fliplr(temp_positions)
    if remove_too_close_atoms:
        atom_positions = _remove_too_close_atoms(
                atom_positions, int(separation)/2)
    return(atom_positions)


def _remove_too_close_atoms(atom_positions, pixel_separation_tolerance):
    index_list = []
    for index0, atom0 in enumerate(atom_positions):
        for index1, atom1 in enumerate(atom_positions):
            too_close = False
            if not ((atom0[0] == atom1[0]) and (atom0[1] == atom1[1])):
                dist = math.hypot(atom0[0]-atom1[0], atom0[1]-atom1[1])
                if pixel_separation_tolerance > dist:
                    if not (index0 in index_list):
                        index_list.append(index1)
    new_atom_positions = []
    for index, atom in enumerate(atom_positions):
        if not index in index_list:
            new_atom_positions.append(atom)
    new_atom_positions = np.array(new_atom_positions)
    return(new_atom_positions)
                

def find_features_by_separation(
        signal,
        separation_range=None,
        separation_step=1,
        threshold_rel=0.02,
        pca=False,
        subtract_background=False,
        normalize_intensity=False,
        ):
    """
    Do peak finding with a varying amount of peak separation
    constrained.

    Inspiration from the program Smart Align by Lewys Jones.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    separation_range : tuple, optional
        Lower and upper end of minimum pixel distance between the
        features.
    separation_step : int, optional

    Returns
    -------
    tuple, (separation_list, peak_list)
    """
    if separation_range is None:
        min_separation = 3
        max_separation = int(np.array(image_data.shape).min()/5)
        separation_range = (min_separation, max_separation)

    separation_list = range(
            separation_range[0],
            separation_range[1],
            separation_step)

    separation_value_list = []
    peak_list = []
    for separation in separation_list:
        peaks = get_atom_positions(
                signal,
                separation=separation,
                threshold_rel=threshold_rel,
                pca=pca,
                normalize_intensity=normalize_intensity,
                subtract_background=subtract_background)

#        peaks = peak_local_max(
#                image=image_data,
#                min_distance=separation,
#                threshold_rel=threshold_rel,
#                indices=True)
        separation_value_list.append(separation)
        peak_list.append(peaks)

    return(separation_value_list, peak_list)


def get_feature_separation(
        signal,
        separation_range=(5, 30),
        separation_step=1,
        pca=False,
        subtract_background=False,
        normalize_intensity=False,
        threshold_rel=0.02,
        ):
    """
    Plot the peak positions on in a HyperSpy signal, as a function
    of peak separation.

    Parameters
    ----------
    signal : HyperSpy signal 2D
    separation_range : tuple, optional, default (5, 30)
    separation_step : int, optional, default 1
    pca : bool, default False
    subtract_background : bool, default False
    normalize_intensity : bool, default False
    threshold_rel : float, default 0.02

    Example
    -------
    >>> import hyperspy.api as hs
    >>> from atomap.atom_finding_refining import get_feature_separation
    >>> s = hs.signals.Signal2D(np.random.random((500, 500))
    >>> s1 = get_feature_separation(s)

    """

    separation_list, peak_list = find_features_by_separation(
            signal=signal,
            separation_range=separation_range,
            separation_step=separation_step,
            threshold_rel=threshold_rel,
            pca=pca,
            normalize_intensity=normalize_intensity,
            subtract_background=subtract_background)

    scale_x = signal.axes_manager[0].scale
    scale_y = signal.axes_manager[1].scale
    offset_x = signal.axes_manager[0].offset
    offset_y = signal.axes_manager[1].offset

    s = hs.stack([signal]*len(separation_list))
    s.axes_manager.navigation_axes[0].offset = separation_list[0]
    s.axes_manager.navigation_axes[0].scale = separation_step
    s.axes_manager.navigation_axes[0].name = "Feature separation, [Pixels]"
    s.axes_manager.navigation_axes[0].unit = "Pixels"

    max_peaks = 0
    for peaks in peak_list:
        if len(peaks) > max_peaks:
            max_peaks = len(peaks)

    marker_list_x = np.ones((len(peak_list), max_peaks))*-100
    marker_list_y = np.ones((len(peak_list), max_peaks))*-100

    for index, peaks in enumerate(peak_list):
        marker_list_x[index, 0:len(peaks)] = (peaks[:, 0]*scale_x)+offset_x
        marker_list_y[index, 0:len(peaks)] = (peaks[:, 1]*scale_y)+offset_y

    marker_list = []
    for i in range(marker_list_x.shape[1]):
        m = hs.markers.point(
                x=marker_list_x[:, i], y=marker_list_y[:, i], color='red')
        marker_list.append(m)

    s.add_marker(marker_list, permanent=True, plot_marker=False)
    return(s)


def find_feature_density(
        image_data,
        separation_range=None,
        separation_step=1,
        plot_figure=False,
        plot_debug_figures=False):
    """
    Do peak finding with a varying amount of peak separation
    constrained. Gives a measure of feature density, and
    what peak separation should be used to find the initial
    sub-lattice.

    Inspiration from the program Smart Align by Lewys Jones.
    """

    separation_list, peak_list = find_features_by_separation(
            image_data=image_data,
            separation_range=separation_range,
            separation_step=separation_step)
    peakN_list = []
    for peaks in peak_list:
        peakN_list.append(len(peaks))

    return(separation_list, peakN_list)


def construct_zone_axes_from_sublattice(
        sublattice, debug_plot=False, zone_axis_para_list=False):
    tag = sublattice._tag
    if sublattice._pixel_separation == 0.0:
        sublattice._pixel_separation = sublattice._get_pixel_separation()
    sublattice._find_nearest_neighbors(nearest_neighbors=15)
    sublattice._make_translation_symmetry()

    if zone_axis_para_list is not False:
        zone_axes = []
        zone_axes_names = []
        for zone_axis_para in zone_axis_para_list:
            if zone_axis_para['number'] < len(
                    sublattice.zones_axis_average_distances):
                index = zone_axis_para['number']
                zone_axes.append(
                        sublattice.zones_axis_average_distances[index])
                zone_axes_names.append(
                        zone_axis_para['name'])
        sublattice.zones_axis_average_distances = zone_axes
        sublattice.zones_axis_average_distances_names = zone_axes_names

    sublattice._generate_all_atom_plane_list()
    sublattice._sort_atom_planes_by_zone_vector()
    sublattice._remove_bad_zone_vectors()
    if debug_plot:
        sublattice.plot_all_atom_planes(fignameprefix=tag+"_atom_plane")


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    y, x = np.ogrid[-centerX:imageSizeX-centerX, -centerY:imageSizeY-centerY]
    mask = x*x + y*y <= radius*radius
    return(mask)


def _make_mask_from_positions(
        position_list,
        radius_list,
        data_shape):
    """
    Parameters
    ----------
    position_list : list of list
        [[x0, y0], [x1, y1]]
    radius_list : list
    data_shape : tuple

    Examples
    --------
    >>> from atomap.atom_finding_refining import _make_mask_from_positions
    >>> pos = [[10, 20], [25, 10]]
    >>> radius = [2, 1]
    >>> mask = _make_mask_from_positions(pos, radius, (40, 40))
    """
    if len(position_list) != len(radius_list):
        raise ValueError(
                "position_list and radius_list must be the same length")
    mask = np.zeros(data_shape)
    for position, radius in zip(position_list, radius_list):
        mask += _make_circular_mask(
                position[0], position[1],
                data_shape[0], data_shape[1],
                radius)
    return(mask)


def _crop_mask_slice_indices(mask):
    """
    Find the outer True values in a mask.
    
    Examples
    --------
    >>> from atomap.atom_finding_refining import _make_mask_from_positions
    >>> from atomap.atom_finding_refining import _crop_mask_slice_indices
    >>> mask = _make_mask_from_positions([[10, 20]], [1], (40, 40))
    >>> x0, x1, y0, y1 = _crop_mask_slice_indices(mask)
    >>> mask_crop = mask[x0:x1, y0:y1] 
    """
    x0 = np.nonzero(mask)[0].min()
    x1 = np.nonzero(mask)[0].max()+1
    y0 = np.nonzero(mask)[1].min()
    y1 = np.nonzero(mask)[1].max()+1
    return(x0, x1, y0, y1)


def _find_background_value(data, method='median', lowest_percentile=0.1):
    """
    Get background value for image data. Intended for Gaussian
    shaped image, with the intensity peak in the middle.
    The median method is preferred to avoid any large outliers
    affecting the background value. Since various acquisition
    artifacts can lead to very low or zero pixel values.

    Parameters
    ----------
    data : Numpy array
    method : 'median', 'mean' or 'minimum'
        If median, the median of the lowest_percentile will
        be used as background. This is the default.
        If mean, the mean of the lowest_percentile is
        used as background.
        If minimum, the lowest value in data will be
        used as background.
    lowest_percentile : number between 0.01 and 1
        The lowest percentile will be used to calculate the
        background, from 1 (0.01) to 100% (1.0). Default 10%

    Returns
    -------
    Float
    """
    if not ((lowest_percentile >= 0.01) and (lowest_percentile <= 1.0)):
        raise ValueError("lowest_percentile must be between 0.01 and 1.0")
    if method == 'minimum':
        background_value = data.min()
    else:
        amount = int(lowest_percentile*data.size)
        if amount == 0:
            amount = 1
        lowest_values = np.sort(data.flatten())[:amount]
        if method == 'median':
            background_value = np.median(lowest_values)
        elif method == 'mean':
            background_value = np.mean(lowest_values)
        else:
            raise ValueError(
                    "method must be 'minimum', 'median' or 'mean'")
    return(background_value)

    
def _find_median_upper_percentile(data, upper_percentile=0.1):
    """
    Get the median of the upper percentile of an image.
    Useful for finding a good initial max value in an image for doing
    2D Gaussian fitting. 
    Median is used to avoid acquisition artifacts leading to too
    high pixel values giving bad initial values.

    Parameters
    ----------
    data : Numpy array
    upper_percentile : number between 0.01 and 1
        The upper percentile will be used to calculate the
        background, from 1 (0.01) to 100% (1.0). Default 10% (0.1).

    Returns
    -------
    Float
    """
    if not ((upper_percentile >= 0.01) and (upper_percentile <= 1.0)):
        raise ValueError("lowest_percentile must be between 0.01 and 1.0")
    amount = int(upper_percentile*data.size)
    if amount == 0:
        amount = 1
    high_value = np.median(np.sort(data.flatten())[-amount:])
    return(high_value)


def _atom_to_gaussian_component(atom):
    g = Gaussian2D(
            centre_x=atom.pixel_x,
            centre_y=atom.pixel_y,
            sigma_x=atom.sigma_x,
            sigma_y=atom.sigma_y,
            rotation=atom.rotation,
            A=1.)
    return(g)


def _make_model_from_atom_list(
        atom_list,
        image_data,
        percent_to_nn=0.40):
    mask = np.zeros_like(image_data)

    position_list, radius_list = [], []
    for atom in atom_list:
        position_list.append((atom.pixel_y, atom.pixel_x))
        radius_list.append(atom.get_closest_neighbor() * percent_to_nn)
    mask = _make_mask_from_positions(
            position_list, radius_list, image_data.shape)
    x0, x1, y0, y1 = _crop_mask_slice_indices(mask)
    mask_crop = mask[x0:x1, y0:y1].astype('bool')
    data_mask_crop = (image_data*mask)[x0:x1, y0:y1]

    upper_value = _find_median_upper_percentile(
            data_mask_crop[mask_crop], upper_percentile=0.03)
    lower_value = _find_background_value(
            data_mask_crop[mask_crop], lowest_percentile=0.03)
    data_mask_crop -= lower_value
    data_mask_crop /= upper_value

    s = Signal2D(data_mask_crop)
    gaussian_list = []
    for atom in atom_list:
        gaussian = _atom_to_gaussian_component(atom)
        gaussian_list.append(gaussian)

    s.axes_manager[0].offset = x0
    s.axes_manager[1].offset = y0
    m = s.create_model()
    m.extend(gaussian_list)

    return(m)


def fit_atom_positions_gaussian(
        atom_list,
        image_data,
        rotation_enabled=True,
        percent_to_nn=0.40,
        centre_free=True,
        debug=False):
    """ If the Gaussian is centered outside the masked area,
    this function returns False"""
    atom = atom_list

    closest_neighbor = atom.get_closest_neighbor()

    slice_size = closest_neighbor * percent_to_nn * 2
    data_slice, x0, y0 = atom._get_image_slice_around_atom(
            image_data, slice_size)

    slice_radius = slice_size/2

    data_slice -= data_slice.min()
    data_slice_max = data_slice.max()
    data = data_slice

    mask = _make_circular_mask(
            slice_radius,
            slice_radius,
            data.shape[0],
            data.shape[1],
            closest_neighbor*percent_to_nn)
    data = deepcopy(data)
    mask = np.invert(mask)
    data[mask] = 0
    g = Gaussian2D(
            centre_x=atom.pixel_x,
            centre_y=atom.pixel_y,
            sigma_x=atom.sigma_x,
            sigma_y=atom.sigma_y,
            rotation=atom.rotation,
            A=data_slice_max)
    
    if centre_free is False:
        g.centre_x.free = False
        g.centre_y.free = False

    if rotation_enabled:
        g.rotation.free = True
    else:
        g.rotation.free = False

    s = Signal2D(data)
    s.axes_manager[0].offset = x0
    s.axes_manager[1].offset = y0
    s = hs.stack([s]*2)
    m = s.create_model()
    m.append(g)
    m.fit()

    if debug:
        atom._plot_gaussian2d_debug(
                slice_radius,
                g,
                data)

    # If the Gaussian centre is located outside the masked region,
    # return False
    dislocation = math.hypot(
            g.centre_x.value-atom.pixel_x,
            g.centre_y.value-atom.pixel_y)
    if dislocation > slice_radius:
        return(False)

    # If sigma aspect ratio is too large, assume the fitting is bad
    max_sigma = max((abs(g.sigma_x.value), abs(g.sigma_y.value)))
    min_sigma = min((abs(g.sigma_x.value), abs(g.sigma_y.value)))
    sigma_ratio = max_sigma/min_sigma
    if sigma_ratio > 5:
        return(False)

    return(g)


def refine_sublattice(
        sublattice,
        refinement_config_list,
        percent_to_nn):

    total_number_of_refinements = 0
    for refinement_config in refinement_config_list:
        total_number_of_refinements += refinement_config[1]

    sublattice._find_nearest_neighbors()

    current_counts = 1
    for refinement_config in refinement_config_list:
        image = refinement_config[0]
        number_of_refinements = refinement_config[1]
        refinement_type = refinement_config[2]
        for index in range(1, number_of_refinements+1):
            print(
                    str(current_counts) + "/" + str(
                        total_number_of_refinements))
            if refinement_type == 'gaussian':
                sublattice.refine_atom_positions_using_2d_gaussian(
                        image,
                        rotation_enabled=False,
                        percent_to_nn=percent_to_nn)
                sublattice.refine_atom_positions_using_2d_gaussian(
                        image,
                        rotation_enabled=True,
                        percent_to_nn=percent_to_nn)
            elif refinement_type == 'center_of_mass':
                sublattice.refine_atom_positions_using_center_of_mass(
                        image,
                        percent_to_nn=percent_to_nn)
            current_counts += 1


# Work in progress
def make_denoised_stem_signal(signal, invert_signal=False):
    signal.change_dtype('float64')
    temp_signal = signal.deepcopy()
    average_background_data = gaussian_filter(
            temp_signal.data, 30, mode='nearest')
    background_subtracted = signal.deepcopy().data -\
        average_background_data
    signal_denoised = hs.signals.Signal(
            background_subtracted-background_subtracted.min())

    signal_denoised.decomposition()
    signal_denoised = signal_denoised.get_decomposition_model(22)
    if not invert_signal:
        signal_denoised_data = 1./signal_denoised.data
        s_abf = 1./s_abf.data
    else:
        signal_den
    signal_denoised = s_abf_modified2/s_abf_modified2.max()
    s_abf_pca = Signal2D(s_abf_data_normalized)


def do_pca_on_signal(signal, pca_components=22):
    signal.change_dtype('float64')
    temp_signal = hs.signals.Signal1D(signal.data)
    temp_signal.decomposition()
    temp_signal = temp_signal.get_decomposition_model(pca_components)
    temp_signal = Signal2D(temp_signal.data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)


def subtract_average_background(signal, gaussian_blur=30):
    signal.change_dtype('float64')
    temp_signal = signal.deepcopy()
    average_background_data = gaussian_filter(
            temp_signal.data, gaussian_blur, mode='nearest')
    background_subtracted = signal.deepcopy().data -\
        average_background_data
    temp_signal = hs.signals.Signal1D(
            background_subtracted-background_subtracted.min())
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
    temp_signal = Signal2D(temp_signal_data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)
