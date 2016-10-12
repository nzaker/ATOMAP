from scipy.ndimage.filters import gaussian_filter
import hyperspy.api as hs
import numpy as np
from skimage.feature import peak_local_max
from atomap_plotting import plot_feature_density
import matplotlib.pyplot as plt


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
                min_distance=int(separation))
    return(peaks)


def find_feature_density(
        image_data,
        separation_range=(3, 40),
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
    if separation_range is None:
        min_separation = 3
        max_separation = int(np.array(image_data.shape).min()/5)
        separation_range = (min_separation, max_separation)

    separation_list = range(
            separation_range[0],
            separation_range[1],
            separation_step)

    separation_value_list = []
    peakN_list = []
    for separation in separation_list:
        peak_list = peak_local_max(image_data, separation)
        separation_value_list.append(separation)
        peakN_list.append(len(peak_list))
        if plot_debug_figures:
            fig, ax = plt.subplots()
            ax.imshow(np.rot90(np.fliplr(image_data)))
            peak_list = peak_list.swapaxes(0, 1)
            ax.scatter(peak_list[0], peak_list[1], color='blue')
            fig.savefig(
                    "feature_density_separation_" +
                    str(separation) + ".png")
            plt.close(fig)

    if plot_figure:
        plot_feature_density(separation_value_list, peakN_list)

    return(separation_value_list, peakN_list)


def construct_zone_axes_from_sub_lattice(sub_lattice):
    tag = sub_lattice.tag
    sub_lattice.find_nearest_neighbors(nearest_neighbors=15)
    sub_lattice._make_nearest_neighbor_direction_distance_statistics(
            debug_figname=tag+"_cat_nn.png")
    sub_lattice._generate_all_atom_plane_list()
    sub_lattice._sort_atom_planes_by_zone_vector()
    sub_lattice.plot_all_atom_planes(fignameprefix=tag+"_atom_plane")


def refine_sub_lattice(
        sub_lattice,
        refinement_config_list,
        percent_distance_to_nearest_neighbor):

    total_number_of_refinements = 0
    for refinement_config in refinement_config_list:
        total_number_of_refinements += refinement_config[1]

    sub_lattice.find_nearest_neighbors()

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
                sub_lattice.refine_atom_positions_using_2d_gaussian(
                        image,
                        rotation_enabled=False,
                        percent_distance_to_nearest_neighbor=
                        percent_distance_to_nearest_neighbor)
                sub_lattice.refine_atom_positions_using_2d_gaussian(
                        image,
                        rotation_enabled=True,
                        percent_distance_to_nearest_neighbor=
                        percent_distance_to_nearest_neighbor)
            elif refinement_type == 'center_of_mass':
                sub_lattice.refine_atom_positions_using_center_of_mass(
                        image,
                        percent_distance_to_nearest_neighbor=
                        percent_distance_to_nearest_neighbor)
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
