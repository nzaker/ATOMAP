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
atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
s_al = atom_lattice.get_sublattice_atom_list_on_image()
s_al.plot()
s_al._plot.signal_plot.figure.savefig(os.path.join(my_path, 'polarization_atom_lattice.png'))

######
sublatticeA = atom_lattice.sublattice_list[0]
sublatticeA.construct_zone_axes()

za0 = sublatticeA.zones_axis_average_distances[0]
za1 = sublatticeA.zones_axis_average_distances[1]
atom_planes0 = sublatticeA.atom_planes_by_zone_vector[za0]
atom_planes1 = sublatticeA.atom_planes_by_zone_vector[za1]

s_ap0 = sublatticeA.get_atom_planes_on_image(atom_planes0)
s_ap1 = sublatticeA.get_atom_planes_on_image(atom_planes1)

s_ap0.plot()
s_ap0._plot.signal_plot.figure.savefig(os.path.join(my_path, 'polarization_atom_plane0.png'))
s_ap1.plot()
s_ap1._plot.signal_plot.figure.savefig(os.path.join(my_path, 'polarization_atom_plane1.png'))

######
sublatticeB = atom_lattice.sublattice_list[1]
s_polarization = sublatticeA.get_polarization_from_second_sublattice(
        za0, za1, sublatticeB)
s_polarization.plot()
s_polarization._plot.signal_plot.figure.savefig(os.path.join(
    my_path, 'polarization_signal_marker.png'), dpi=150)
