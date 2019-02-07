import os
import numpy as np
import matplotlib.pyplot as plt
import atomap.api as am
from atomap.tools import remove_atoms_from_image_using_2d_gaussian

my_path = os.path.join(os.path.dirname(__file__), 'finding_atom_lattices')
if not os.path.exists(my_path):
    os.makedirs(my_path)

# Loading signals
s_ADF = am.dummy_data.get_two_sublattice_signal()
s_ABF = am.dummy_data.get_perovskite110_ABF_signal(image_noise=True)

# Plot ABF signal
s_ABF.plot()
s_ABF._plot.signal_plot.figure.savefig(os.path.join(my_path, 's_ABF.png'))

# Finding sublattice_A from ADF
A_positions = am.get_atom_positions(s_ADF, separation=15)
sublattice_A = am.Sublattice(A_positions, image=s_ADF.data, color='r')
sublattice_A.find_nearest_neighbors()
sublattice_A.refine_atom_positions_using_center_of_mass()
sublattice_A.refine_atom_positions_using_2d_gaussian()
sublattice_A.construct_zone_axes()

# Finding sublattice_B from ADF
zone_axis_001 = sublattice_A.zones_axis_average_distances[1]
B_positions = sublattice_A.find_missing_atoms_from_zone_vector(zone_axis_001)
image_without_A = remove_atoms_from_image_using_2d_gaussian(
        sublattice_A.image,
        sublattice_A)
sublattice_B = am.Sublattice(B_positions, image_without_A, color='blue')
sublattice_B.construct_zone_axes()
sublattice_B.refine_atom_positions_using_center_of_mass()
sublattice_B.refine_atom_positions_using_2d_gaussian()

# Finding sublattice_A2 in ABF for subtraction
xy = sublattice_A.atom_positions
sublattice_A2 = am.Sublattice(xy, image=np.divide(1, s_ABF.data), color='r')
sublattice_A2.find_nearest_neighbors()
sublattice_A2.refine_atom_positions_using_center_of_mass()
sublattice_A2.refine_atom_positions_using_2d_gaussian()
sublattice_A2.construct_zone_axes()

# Finding sublattice_B2 in ABF for subtraction
image_without_A2 = remove_atoms_from_image_using_2d_gaussian(
        sublattice_A2.image,
        sublattice_A2)
xy = sublattice_B.atom_positions
sublattice_B2 = am.Sublattice(xy, image=image_without_A2, color='b')
sublattice_B2.find_nearest_neighbors()
sublattice_B2.refine_atom_positions_using_center_of_mass()
sublattice_B2.refine_atom_positions_using_2d_gaussian()
sublattice_B2.construct_zone_axes()

sublattice_B2.plot_planes()
plt.gcf().savefig(os.path.join(my_path, 'sublattice_B2.png'))

# Finding Oxygen
zone_axis_002 = sublattice_B2.zones_axis_average_distances[0]
O_positions = sublattice_B2.find_missing_atoms_from_zone_vector(zone_axis_002)
image_without_AB = remove_atoms_from_image_using_2d_gaussian(
        sublattice_B2.image,
        sublattice_B2)

sublattice_O = am.Sublattice(O_positions, image_without_AB, color='g')
sublattice_O.construct_zone_axes()
sublattice_O.refine_atom_positions_using_center_of_mass()
sublattice_O.refine_atom_positions_using_2d_gaussian()

s = sublattice_O.get_atom_list_on_image()
s.plot()
s._plot.signal_plot.ax.set_xlim(70, 130)
s._plot.signal_plot.ax.set_ylim(50, 100)
s._plot.signal_plot.figure.savefig(
        os.path.join(my_path, 'oxygen_positions.png'))

atom_lattice = am.Atom_Lattice(
        image=s_ABF.data, name='ABO3',
        sublattice_list=[sublattice_A, sublattice_B, sublattice_O])

atom_lattice.plot()
plt.gcf().savefig(os.path.join(my_path, 'ABO3.png'))
s = atom_lattice.get_sublattice_atom_list_on_image(image=s_ADF.data)
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(my_path, 'ABO3-ADF.png'))
