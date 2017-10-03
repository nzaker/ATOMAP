import atomap.api as am
import hyperspy.api as hs
import numpy as np
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
from atomap.plotting import plot_vector_field
import matplotlib.pyplot as plt

def plot_fantasite():
    s = am.dummy_data.get_fantasite()
    s.plot()
    plt.gcf().savefig('atom_lattice_image.png')

def plot_elli_maps(sublattice_A):
    sublattice_A.plot_ellipticity_map(cmap='viridis',vmin=0.95,vmax=1.3)
    plt.gcf().savefig('ellipticity_map_A.png')

    sublattice_B.plot_ellipticity_map(cmap='viridis',vmin=0.95,vmax=1.3)
    plt.gcf().savefig('ellipticity_map_B.png')

    sublattice_A.plot_ellipticity_vectors()
    plt.gcf().savefig('ellipticity_vectors.png')


def plot_monolayer_map(sublattice_B):
    z_1 =  sublattice_B.zones_axis_average_distances[1]
    s_monolayer = sublattice_B.get_monolayer_distance_map()
    s_monolayer.plot(cmap='viridis')
    plt.gcf().savefig('Sublattice_B_monolayer_distance_a.png')

def plot_angle_maps(sublattice_B):
    z1 =  sublattice_B.zones_axis_average_distances[0]
    z2 =  sublattice_B.zones_axis_average_distances[1]
    x, y, a = sublattice_B.get_atom_angles_from_zone_vector(z1, z2, degrees=True)
    s_angle = sublattice_B._get_property_map(x, y, a)
    s_angle.plot(cmap='magma')
    plt.gcf().savefig('Angle_map.png')


def elli_line(sublattice_A):
    zone = sublattice_A.zones_axis_average_distances[1]
    plane = sublattice_A.atom_planes_by_zone_vector[zone][8]
    s_elli_line = sublattice_A.get_ellipticity_line_profile(plane)

def monlolayer_line(sublattice_B):
    zone = sublattice_B.zones_axis_average_distances[1]
    plane = sublattice_B.atom_planes_by_zone_vector[zone][0]
    s_monolayer_line = sublattice_B.get_monolayer_distance_line_profile(zone,plane)
    s_monolayer_line.plot()
    plt.gcf().savefig('line_monolayer.png')

def dd_line(sublattice_B):
    zone = sublattice_B.zones_axis_average_distances[1]
    plane = sublattice_B.atom_planes_by_zone_vector[zone][21]
    s_dd = sublattice_B.get_atom_distance_difference_line_profile(zone,plane)
    s_dd.plot()
    plt.gcf().savefig('line_dd.png')

def dd_map_B(sublattice_B,sublattice_A):
    zone = sublattice_B.zones_axis_average_distances[1]
    s_dd = sublattice_B.get_atom_distance_difference_map([zone],add_zero_value_sublattice=sublattice_A)
    s_dd.plot(cmap='viridis')
    plt.gcf().savefig('sublatticeB_dd_map_1.png')

def atom_lattice(atom_lattice):
    atom_lattice.plot()
    plt.gcf().savefig('atom_lattice.png')


atom_lattice = am.load_atom_lattice_from_hdf5('fantasite.hdf5')

A = atom_lattice.sublattice_list[0]
B = atom_lattice.sublattice_list[1]



