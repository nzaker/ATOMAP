import atomap.api as am
import hyperspy.api as hs
import numpy as np
from atomap.tools import remove_atoms_from_image_using_2d_gaussian

s = am.dummy_data.get_fantasite()

#s_peaks = am.get_feature_separation(s)

A_positions = am.get_atom_positions(s,separation=12)
sublattice_A = am.Sublattice(A_positions,image=s.data, color='r', name='A')
sublattice_A.find_nearest_neighbors()
sublattice_A.refine_atom_positions_using_center_of_mass()
sublattice_A.refine_atom_positions_using_2d_gaussian()
sublattice_A.construct_zone_axes()

sublattice_A.get_all_atom_planes_by_zone_vector().plot()

direction_001 = sublattice_A.zones_axis_average_distances[1]
B_positions = sublattice_A._find_missing_atoms_from_zone_vector(direction_001)
image_without_A = remove_atoms_from_image_using_2d_gaussian(sublattice_A.image, sublattice_A)

sublattice_B = am.Sublattice(B_positions, image_without_A, color='blue', name='B')
sublattice_B.construct_zone_axes()
sublattice_B.refine_atom_positions_using_center_of_mass()
sublattice_B.refine_atom_positions_using_2d_gaussian()

atom_lattice = am.Atom_Lattice(
        image=s.data, name='fantasite',
        sublattice_list=[sublattice_A, sublattice_B])
atom_lattice.save("fantasite.hdf5", overwrite=True)

