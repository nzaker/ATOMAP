from scipy.ndimage.filters import gaussian_filter
import hyperspy.api as hs
import numpy as np
from skimage.feature import peak_local_max
from copy import deepcopy
import math


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
