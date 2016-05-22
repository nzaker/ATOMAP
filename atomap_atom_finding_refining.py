def construct_zone_axes_from_atom_lattice(atom_lattice):
    tag = atom_lattice.tag
    atom_lattice.find_nearest_neighbors(nearest_neighbors=15)
    atom_lattice._make_nearest_neighbor_direction_distance_statistics(
            debug_figname=tag+"_cat_nn.png")
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
                        rotation_enabled=False,
                        percent_distance_to_nearest_neighbor=\
                        percent_distance_to_nearest_neighbor)
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

# DENNE ER UFERDIG
def make_denoised_stem_signal(signal, invert_signal=False):
    signal.change_dtype('float64')
    temp_signal = signal.deepcopy()
    average_background_data = gaussian_filter(temp_signal.data, 30, mode='nearest')
    background_subtracted = signal.deepcopy().data - average_background_data
    signal_denoised = hs.signals.Signal(background_subtracted-background_subtracted.min())

    signal_denoised.decomposition()
    signal_denoised = signal_denoised.get_decomposition_model(22)
    if not invert_signal:
        signal_denoised_data = 1./signal_denoised.data
        s_abf = 1./s_abf.data
    else:
        signal_den
    signal_denoised = s_abf_modified2/s_abf_modified2.max()
    s_abf_pca = hs.signals.Image(s_abf_data_normalized)

def do_pca_on_signal(signal, pca_components=22):
    signal.change_dtype('float64')
    temp_signal = hs.signals.Signal(signal.data)
    temp_signal.decomposition()
    temp_signal = temp_signal.get_decomposition_model(pca_components)
    temp_signal = hs.signals.Image(temp_signal.data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)

def subtract_average_background(signal, gaussian_blur=30):
    signal.change_dtype('float64')
    temp_signal = signal.deepcopy()
    average_background_data = gaussian_filter(
            temp_signal.data, gaussian_blur, mode='nearest')
    background_subtracted = signal.deepcopy().data - average_background_data
    temp_signal = hs.signals.Signal(background_subtracted-background_subtracted.min())
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
    temp_signal = hs.signals.Image(temp_signal_data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return(temp_signal)

