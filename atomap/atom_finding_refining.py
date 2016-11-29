from scipy.ndimage.filters import gaussian_filter
import hyperspy.api as hs
import numpy as np
from skimage.feature import peak_local_max
from atomap.plotting import plot_feature_density
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


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
    >>>> peaks = get_peak2d_skimage(s, 5)
    >>>> peak_x = peaks[0][:,0]
    >>>> peak_y = peaks[0][:,1]
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


def plot_feature_separation(
        signal,
        separation_range=(5, 30),
        separation_step=1):
    """
    Plot peak positions as a function of peak separation.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    separation_range : tuple, optional
    separation_step : int, optional

    Examples
    --------
    >>>> import hyperspy.api as hs
    >>>> from atomap.atom_finding_refining import plot_feature_separation
    >>>> s = hs.load("stem_adf_data.hdf5")
    >>>> plot_feature_separation(s)

    Using all the parameters
    >>>> plot_feature_separation(s, separation_range=(10,50), separation_step=3)
    """
    image_data = signal.data

    # skimage.feature.peak_local_max used in find_features_by_separation
    # only support 32-bit or higher
    if image_data.dtype is np.dtype('float16'):
        image_data = image_data.astype('float32')
    if image_data.dtype is np.dtype('int8'):
        image_data = image_data.astype('int32')
    if image_data.dtype is np.dtype('int16'):
        image_data = image_data.astype('int32')
    separation_list, peak_list = find_features_by_separation(
            image_data=image_data,
            separation_range=separation_range,
            separation_step=separation_step)
    for index, (separation, peaks) in enumerate(
            zip(separation_list, peak_list)):
        fig = Figure(figsize=(7, 7))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(image_data)
        ax.scatter(peaks[:, 1], peaks[:, 0])
        ax.set_xlim(0, image_data.shape[1])
        ax.set_ylim(0, image_data.shape[0])
        ax.set_axis_off()
        ax.set_title("Peak separation, " + str(separation) + " pixels")
        fig.tight_layout()
        fig.savefig("peak_separation_" + str(separation).zfill(3))


def plot_feature_separation_hyperspy_signal(
        image_data,
        separation_range=(5, 30),
        separation_step=1):
    """
    Plot the peak positions on in a HyperSpy signal.

    Note: this is currently not working.
    """
    separation_list, peak_list = find_features_by_separation(
            image_data=image_data,
            separation_range=separation_range,
            separation_step=separation_step)

    s = hs.signals.Signal2D([image_data]*len(separation_list))
    peak_list = np.array(peak_list)

    max_peaks = 0
    for peaks in peak_list:
        if len(peaks) > max_peaks:
            max_peaks = len(peaks)

    marker_list_x = np.ones((len(peak_list), max_peaks))
    marker_list_y = np.ones((len(peak_list), max_peaks))

#    print(peak_list.shape)
#    for index, peaks in enumerate(peak_list):
#        marker_list_x[index, 0:len(peaks)] = copy.deepcopy(peaks[:,1])
#        marker_list_y[index, 0:len(peaks)] = copy.deepcopy(peaks[:,0])
#    print(marker_list_x.shape)
#    print(marker_list_y.shape)

#    s.axes_manager.navigation_axes[0].offset = separation_list[0]
#    s.axes_manager.navigation_axes[0].scale = separation_step

#    m = hs.plot.markers.point(
#            x=marker_list_x, y=marker_list_y, color='red')
#    s.add_marker(m)

    m = hs.plot.markers.point(
            x=peak_list[:, 1], y=peak_list[:, 0], color='red')
    s.add_marker(m)

#    for index, (marker_x, marker_y) in enumerate(zip(marker_list_x, marker_list_y)):
#        m = hs.plot.markers.point(
#                x=marker_x, y=marker_y, color='red')
#        s.add_marker(m,
#                plot_on_signal=True,
#                plot_marker=True)
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


def construct_zone_axes_from_sublattice(sublattice):
    tag = sublattice.tag
    sublattice.find_nearest_neighbors(nearest_neighbors=15)
    sublattice._make_nearest_neighbor_direction_distance_statistics(
            debug_figname=tag+"_cat_nn.png")
    sublattice._generate_all_atom_plane_list()
    sublattice._sort_atom_planes_by_zone_vector()
    sublattice.plot_all_atom_planes(fignameprefix=tag+"_atom_plane")


def refine_sublattice(
        sublattice,
        refinement_config_list,
        percent_to_nn):

    total_number_of_refinements = 0
    for refinement_config in refinement_config_list:
        total_number_of_refinements += refinement_config[1]

    sublattice.find_nearest_neighbors()

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
