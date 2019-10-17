import os
import numpy as np
import atomap.api as am
import atomap.testing_tools as tt
import atomap.analysis_tools as an
import atomap.plotting as pl

my_path = os.path.join(os.path.dirname(__file__), 'severalphases')
if not os.path.exists(my_path):
    os.makedirs(my_path)

######
s = am.dummy_data.get_precipitate_signal()
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(my_path, 'precipitate_signal.png'))

######
peaks = am.get_atom_positions(s, 8)
verts = [[250, 100], [100, 250], [250, 400], [400, 250]]
peaks_p = am.select_atoms_with_gui(s, peaks, verts)

sublattice_p = am.Sublattice(peaks_p, s)
s_p = sublattice_p.get_atom_list_on_image()
s_p.plot()
s_p._plot.signal_plot.figure.savefig(os.path.join(my_path, 'precipitate_sublattice.png'))

######
peaks_m = am.select_atoms_with_gui(s, peaks, verts, invert_selection=True)
sublattice_m = am.Sublattice(peaks_m, s, color='blue')
s_m = sublattice_m.get_atom_list_on_image()
s_m.plot()
s_m._plot.signal_plot.figure.savefig(os.path.join(my_path, 'matrix_sublattice.png'))

######
atom_lattice = am.Atom_Lattice(s, sublattice_list=[sublattice_p, sublattice_m])
s_a = atom_lattice.get_sublattice_atom_list_on_image()
s_a.plot()
s_a._plot.signal_plot.figure.savefig(os.path.join(my_path, 'atom_lattice.png'))

######
i_points, i_record, p_record = atom_lattice.integrate_column_intensity()
i_record.plot()
i_record._plot.signal_plot.figure.savefig(os.path.join(my_path, 'atom_lattice_integrate.png'))

i_record_crop = i_record.isig[30:-30, 30:-30]
i_record_crop.plot()
i_record_crop._plot.signal_plot.figure.savefig(os.path.join(my_path, 'atom_lattice_integrate_crop.png'))

######
sublattice_p.construct_zone_axes()
sublattice_p.refine_atom_positions_using_center_of_mass()
sublattice_p.refine_atom_positions_using_2d_gaussian()

za0 = sublattice_p.zones_axis_average_distances[0]
s_mono0 = sublattice_p.get_monolayer_distance_map([za0])
s_mono0.plot()
s_mono0._plot.signal_plot.figure.savefig(os.path.join(my_path, 'precipitate_monolayer0.png'))
