from tqdm import trange
from atomap.atom_finding_refining import\
        construct_zone_axes_from_sublattice, fit_atom_positions_gaussian
from atomap.plotting import _make_atom_position_marker_list
from atomap.tools import array2signal2d


class Atom_Lattice():

    def __init__(
            self,
            image=None,
            name="",
            sublattice_list=None,
            ):
        """
        Parameters
        ----------
        image : 2D NumPy array, optional
        sublattice_list : list of sublattice object, optional
        """
        if sublattice_list is None:
            self.sublattice_list = []
        else:
            self.sublattice_list = sublattice_list
        if image is None:
            self.image0 = None
        else:
            self.image0 = image
        self.name = name
        self._pixel_separation = 10
        self._original_filename = ''

    def __repr__(self):
        return '<%s, %s (sublattice(s): %s)>' % (
            self.__class__.__name__,
            self.name,
            len(self.sublattice_list),
            )

    def get_sublattice(self, sublattice_id):
        """
        Get a sublattice object from either sublattice index
        or name.
        """
        if isinstance(sublattice_id, str):
            for sublattice in self.sublattice_list:
                if sublattice.name == sublattice_id:
                    return(sublattice)
        elif isinstance(sublattice_id, int):
            return(self.sublattice_list[sublattice_id])
        raise ValueError('Could not find sublattice ' + str(sublattice_id))

    def _construct_zone_axes_for_sublattices(self, sublattice_list=None):
        if sublattice_list is None:
            sublattice_list = self.sublattice_list
        for sublattice in sublattice_list:
            construct_zone_axes_from_sublattice(sublattice)

    def get_sublattice_atom_list_on_image(
            self,
            image=None,
            add_numbers=False,
            markersize=20):
        if image is None:
            image = self.image0
        marker_list = []
        scale = self.sublattice_list[0].pixel_size
        for sublattice in self.sublattice_list:
            marker_list.extend(_make_atom_position_marker_list(
                    sublattice.atom_list,
                    scale=scale,
                    color=sublattice._plot_color,
                    markersize=markersize,
                    add_numbers=add_numbers))
        signal = array2signal2d(image, scale)
        signal.add_marker(marker_list, permanent=True, plot_marker=False)

        return signal

    def save(self, filename=None, overwrite=False):
        """
        Save the AtomLattice object as an HDF5 file.
        This will store the information about the individual atomic
        positions, like the pixel position, sigma and rotation.
        The input image(s) and modified version of these will also
        be saved.

        Parameters
        ----------
        filename : string, optional
        overwrite : bool, default False

        Examples
        --------
        >>> from numpy.random import random
        >>> import atomap.api as am
        >>> sl = am.dummy_data.get_simple_cubic_sublattice()
        >>> atom_lattice = am.Atom_Lattice(random((9, 9)), "test", [sl])
        >>> atom_lattice.save("test.hdf5", overwrite=True)

        Loading the atom lattice:

        >>> import atomap.api as am
        >>> atom_lattice1 = am.load_atom_lattice_from_hdf5("test.hdf5")
        """
        from atomap.io import save_atom_lattice_to_hdf5
        if filename is None:
            filename = self.name + "_atom_lattice.hdf5"
        save_atom_lattice_to_hdf5(self, filename=filename, overwrite=overwrite)

    def plot(self,
             image=None,
             add_numbers=False,
             markersize=20,
             **kwargs):
        """
        Plot all atom positions for all sub lattices on the image data.

        The Atom_Lattice.image0 is used as the image. For the sublattices,
        sublattice._plot_color is used as marker color. This color is set
        when the sublattice is initialized, but it can also be changed.

        Parameters
        ----------
        image : 2D NumPy array, optional
        add_numbers : bool, default False
            Plot the number of the atom beside each atomic position in the
            plot. Useful for locating misfitted atoms.
        markersize : number, default 20
            Size of the atom position markers
        **kwargs
            Addition keywords passed to HyperSpy's signal plot function.

        Examples
        --------
        >>> import atomap.api as am
        >>> import atomap.testing_tools as tt
        >>> test_data = tt.MakeTestData(50, 50)
        >>> import numpy as np
        >>> test_data.add_atom_list(np.arange(5, 45, 5), np.arange(5, 45, 5))
        >>> atom_lattice = test_data.atom_lattice
        >>> atom_lattice.plot()

        Change sublattice colour and color map

        >>> atom_lattice.sublattice_list[0]._plot_color = 'green'
        >>> atom_lattice.plot(cmap='viridis')

        See also
        --------
        get_sublattice_atom_list_on_image : get HyperSpy signal with atom
            positions as markers. More customizability.
        """
        signal = self.get_sublattice_atom_list_on_image(
            image=image,
            add_numbers=add_numbers,
            markersize=markersize)
        signal.plot(**kwargs, plot_markers=True)


class Dumbbell_Lattice(Atom_Lattice):

    def refine_position_gaussian(self, image=None):
        if image is None:
            image = self.image0
        n_tot = len(self.sublattice_list[0].atom_list)
        for i_atom in trange(n_tot, desc="Gaussian fitting"):
            atom_list = []
            for sublattice in self.sublattice_list:
                atom_list.append(sublattice.atom_list[i_atom])
            fit_atom_positions_gaussian(
                    atom_list,
                    image)
