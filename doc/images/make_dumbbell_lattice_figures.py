import os
import numpy as np
import atomap.api as am
import atomap.initial_position_finding as ipf

my_path = os.path.join(os.path.dirname(__file__), 'makedumbbelllattice')
if not os.path.exists(my_path):
    os.makedirs(my_path)

#####
s = am.dummy_data.get_dumbbell_heterostructure_signal()
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(my_path, 'dummy_data.png'))

#####
s_peaks = am.get_feature_separation(s, separation_range=(2, 6))
s_peaks.plot()
s_peaks._plot.signal_plot.figure.savefig(os.path.join(my_path, 'feature_separation_all_atoms.png'))

#####
atom_positions = am.get_atom_positions(s, separation=2)
dumbbell_vector = ipf.find_dumbbell_vector(atom_positions)

#####
s_peaks = am.get_feature_separation(s, separation_range=(5, 20))
s_peaks.axes_manager.indices = (3,)
s_peaks.plot()
s_peaks._plot.signal_plot.figure.savefig(os.path.join(my_path, 'feature_separation_dumbbell.png'))

#####
dumbbell_positions = am.get_atom_positions(s, separation=8)
dumbbell_lattice = ipf.make_atom_lattice_dumbbell_structure(s, dumbbell_positions, dumbbell_vector)
s_ap = dumbbell_lattice.get_sublattice_atom_list_on_image()
s_ap.plot()
s_ap._plot.signal_plot.figure.savefig(os.path.join(my_path, 'dumbbell_lattice_initial.png'))

#####
i_points, i_record, p_record = dumbbell_lattice.integrate_column_intensity()
i_record.plot()
i_record._plot.signal_plot.figure.savefig(os.path.join(my_path, 'integrated_intensity.png'))

#####
sublattice0 = dumbbell_lattice.sublattice_list[0]
sublattice0.construct_zone_axes()
out_of_plane_direction = sublattice0.zones_axis_average_distances[2]
interface_plane = sublattice0.atom_planes_by_zone_vector[out_of_plane_direction][15]
s_out_of_plane_map = sublattice0.get_monolayer_distance_map([out_of_plane_direction, ], atom_plane_list=[interface_plane])
s_out_of_plane_map.plot()
s_out_of_plane_map._plot.signal_plot.figure.savefig(os.path.join(my_path, 'sublattice0_out_of_plane_map.png'))

#####
s_out_of_plane_line_profile = sublattice0.get_monolayer_distance_line_profile(out_of_plane_direction, atom_plane=interface_plane)
s_out_of_plane_line_profile.plot()
s_out_of_plane_line_profile._plot.signal_plot.figure.savefig(os.path.join(my_path, 'sublattice0_out_of_plane_line_profile.png'))
