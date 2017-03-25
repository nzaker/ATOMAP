from scipy.ndimage.filters import gaussian_filter
import hyperspy.api as hs
import numpy as np
from skimage.feature import peak_local_max
from copy import deepcopy


def get_peak2d_skimage(image, separation):
    """
    Find the most intense features in a HyperSpy signal, where the
    features has to be separated by a minimum distance.

    Will work with image stacks.

    Parameters
    ----------
    image : HyperSpy 2D signal
        Can be in the form of an image stack.
    separation : number
        Minimum separation between the features.

    Returns
    -------
    Numpy array, list of the most intense peaks.

    Example
    -------
    If s is a single image
    >>> peaks = get_peak2d_skimage(s, 5)
    >>> peak_x = peaks[0][:,0]
    >>> peak_y = peaks[0][:,1]
    """
    peaks = np.zeros([image.axes_manager.navigation_size+1, ], dtype=object)
    for z, indices in zip(
            image._iterate_signal(),
            image.axes_manager._array_indices_generator()):
            peaks[indices] = peak_local_max(
                z,
                min_distance=int(separation))
    return(peaks)


def find_features_by_separation(
        image_data,
        separation_range=None,
        separation_step=1):
    """
    Do peak finding with a varying amount of peak separation
    constrained.

    Inspiration from the program Smart Align by Lewys Jones.

    Parameters
    ----------
    image_data : Numpy 2D array
    separation_range : tuple, optional
        Lower and upper end of minimum pixel distance between the
        features.
    separation_step : int, optional

    Returns
    -------
    tuple, (separation_list, peak_list)
    """
    if image_data.dtype is np.dtype('float16'):
        image_data = image_data.astype('float32')
    if image_data.dtype is np.dtype('int8'):
        image_data = image_data.astype('int32')
    if image_data.dtype is np.dtype('int16'):
        image_data = image_data.astype('int32')

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
        peaks = peak_local_max(image_data, separation)
        separation_value_list.append(separation)
        peak_list.append(peaks)

    return(separation_value_list, peak_list)


def get_feature_separation(
        signal,
        separation_range=(5, 30),
        separation_step=1,
        pca=False,
        subtract_background=False,
        normalize_intensity=False):
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

    Example
    -------
    >>> import hyperspy.api as hs
    >>> from atomap.atom_finding_refining import get_feature_separation
    >>> s = hs.signals.Signal2D(np.random.random((500, 500))
    >>> s1 = get_feature_separation(s)

    """
    if pca:
        signal = do_pca_on_signal(signal)
    if subtract_background:
        signal = subtract_average_background(signal)
    if normalize_intensity:
        signal = normalize_signal(signal)

    image_data = deepcopy(signal.data)
    separation_list, peak_list = find_features_by_separation(
            image_data=image_data,
            separation_range=separation_range,
            separation_step=separation_step)

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
        marker_list_x[index, 0:len(peaks)] = (peaks[:, 1]*scale_x)+offset_x
        marker_list_y[index, 0:len(peaks)] = (peaks[:, 0]*scale_y)+offset_y

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
    sublattice._find_nearest_neighbors(nearest_neighbors=15)
    sublattice._make_nearest_neighbor_direction_distance_statistics(
            debug_plot=debug_plot)

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
    s_abf_pca = hs.signals.Signal2D(s_abf_data_normalized)


def do_pca_on_signal(signal, pca_components=22):
    signal.change_dtype('float64')
    temp_signal = hs.signals.Signal1D(signal.data)
    temp_signal.decomposition()
    temp_signal = temp_signal.get_decomposition_model(pca_components)
    temp_signal = hs.signals.Signal2D(temp_signal.data)
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
    temp_signal = hs.signals.Signal2D(temp_signal_data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)
