# Rename to Atom_Lattice
class Material_Structure():
    def __init__(self):
        self.atom_lattice_list = []
        self.adf_image = None
        self.inverted_abf_image = None

    def construct_zone_axes_for_atom_lattices(self, atom_lattice_list=None):
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            construct_zone_axes_from_atom_lattice(atom_lattice)

    def plot_all_atom_lattices(self, image=None, markersize=1, figname="all_atom_lattice.jpg"):
        if image == None:
            image = self.adf_image
        fig, ax = plt.subplots(figsize=(10,10))
        cax = ax.imshow(self.adf_image)
        for atom_lattice in self.atom_lattice_list:
            color = atom_lattice.plot_color
            for atom in atom_lattice.atom_list:
                ax.plot(atom.pixel_x, atom.pixel_y, 'o', markersize=markersize, color=color)
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(figname)

# To be removed
#    def plot_all_atom_lattices_abf(self, markersize=1, figname="all_atom_lattice_abf.jpg"):
#        plt.ioff()
#        fig, ax = plt.subplots(figsize=(10,10))
#        cax = ax.imshow(self.abf_image)
#        for atom_lattice in self.atom_lattice_list:
#            color = atom_lattice.plot_color
#            for atom in atom_lattice.atom_list:
#                ax.plot(atom.pixel_x, atom.pixel_y, 'o', markersize=markersize, color=color)
#        ax.set_ylim(0, self.abf_image.shape[0])
#        ax.set_xlim(0, self.abf_image.shape[1])
#        fig.savefig(figname)

    def plot_atom_distance_maps_for_zone_vectors_and_lattices(
            self,
            atom_lattice_list=None,
            interface_row=None,
            max_number_of_zone_vectors=1):
        plt.ioff()
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            atom_lattice.plot_distance_map_for_all_zone_vectors(
                atom_row_marker=interface_row,
                atom_list=atom_lattice.atom_list,
                max_number_of_zone_vectors=max_number_of_zone_vectors)

    def plot_atom_distance_difference_maps_for_zone_vectors_and_lattices(
            self,
            atom_list_as_zero=None,
            atom_lattice_list=None):
        plt.ioff()
        if atom_lattice_list == None:
            atom_lattice_list = self.atom_lattice_list
        for atom_lattice in atom_lattice_list:
            atom_lattice.plot_distance_difference_map_for_all_zone_vectors(
                    atom_list_as_zero=atom_list_as_zero)
 
