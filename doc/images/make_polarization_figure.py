import os
import numpy as np
import atomap.api as am
import atomap.testing_tools as tt
import atomap.analysis_tools as an
import atomap.plotting as pl

my_path = os.path.join(os.path.dirname(__file__), 'makepolarization')
if not os.path.exists(my_path):
    os.makedirs(my_path)

######
signal = am.dummy_data.get_polarization_film_signal()
signal.plot()
signal._plot.signal_plot.figure.savefig(os.path.join(my_path, 'polarization_signal.png'))

######
atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
sublatticeA = atom_lattice.sublattice_list[0]
sublatticeB = atom_lattice.sublattice_list[1]
sublatticeA.construct_zone_axes()
sublatticeB.construct_zone_axes()

za0 = sublatticeA.zones_axis_average_distances[0]
za1 = sublatticeA.zones_axis_average_distances[1]

middle_position_list = sublatticeA.get_middle_position_list(za0, za1)
vector_list = an.get_vector_shift_list(sublatticeB, middle_position_list)
marker_list = pl.vector_list_to_marker_list(vector_list, color='cyan', scale=1.)

s = sublatticeA.get_atom_list_on_image()
s.add_marker(marker_list, permanent=True, plot_signal=False)
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(my_path, 'polarization_signal_marker.png'))
