import numpy as np
import matplotlib.pyplot as plt
import math
import atomap.tools as to


def get_neighbor_middle_position(atom, za0, za1):
    """Find the middle point between four neighboring atoms.

    The neighbors are found by moving one step along the atom planes
    belonging to za0 and za1.

    So atom planes must be constructed first.

    Parameters
    ----------
    atom : Atom_Position object
    za0 : tuple
    za1 : tuple

    Return
    ------
    middle_position : tuple
        If the atom is at the edge by being the last atom in the
        atom plane, False is returned.

    Examples
    --------
    >>> import atomap.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> atom = sublattice.atom_list[33]
    >>> middle_position = an.get_neighbor_middle_position(atom, za0, za1)

    """
    atom00 = atom
    atom01 = atom.get_next_atom_in_zone_vector(za0)
    atom10 = atom.get_next_atom_in_zone_vector(za1)
    middle_position = False
    if not (atom01 is False):
        if not (atom10 is False):
            atom11 = atom10.get_next_atom_in_zone_vector(za0)
            if not (atom11 is False):
                middle_position = to.get_point_between_four_atoms((
                        atom00, atom01, atom10, atom11))
    return middle_position


def get_middle_position_list(sublattice, za0, za1):
    """Find the middle point between all four neighboring atoms.

    The neighbors are found by moving one step along the atom planes
    belonging to za0 and za1.

    So atom planes must be constructed first.

    Parameters
    ----------
    sublattice : Sublattice object
    za0 : tuple
    za1 : tuple

    Return
    ------
    middle_position_list : list

    Examples
    --------
    >>> import atomap.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> middle_position_list = an.get_middle_position_list(
    ...     sublattice, za0, za1)

    """
    middle_position_list = []
    for atom in sublattice.atom_list:
        middle_pos = get_neighbor_middle_position(atom, za0, za1)
        if not (middle_pos is False):
            middle_position_list.append(middle_pos)
    return middle_position_list


def get_vector_shift_list(sublattice, position_list):
    """Find the atom shifts from a central position.

    Useful for finding polarization in B-cations in a perovskite structure.

    Parameters
    ----------
    sublattice : Sublattice object
    position_list : list
        [[x0, y0], [x1, y1], ...]

    Returns
    -------
    vector_list : list
        In the form [[x0, y0, dx0, dy0], [x1, y1, dx1, dy1]...]

    Example
    -------
    >>> import atomap.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> middle_position_list = an.get_middle_position_list(
    ...     sublattice, za0, za1)
    >>> vector_list = an.get_vector_shift_list(
    ...     sublattice, middle_position_list)

    """
    vector_list = []
    for position in position_list:
        dist = np.hypot(
                np.array(sublattice.x_position) - position[0],
                np.array(sublattice.y_position) - position[1])
        atom_b = sublattice.atom_list[dist.argmin()]
        vector = (position[0], position[1],
                  position[0] - atom_b.pixel_x,
                  position[1] - atom_b.pixel_y)
        vector_list.append(vector)
    return vector_list


def pair_distribution_function(
        image, atom_positions, plot=True, n_bins=200, rel_range=0.5):
    """
    Returns a two dimensional pair distribution function (PDF) from an image of
    atomic columns.

    The intensity of peaks in the PDF is corrected to account for missing
    information (i.e. the fact atoms are present outside of the field of view)
    and differences in area at different distances.

    Parameters
    ----------
    image : 2D Hyperspy Signal object
    atom_positions : numpy array
        A numpy array of [x,y] atom positions.
    plot : bool
        If True will plot the PDF.
    n_bins : int
        Number of bins to use for the PDF.
    rel_range : float
        The range of the PDF as a fraction of the field of view of the image.

    Returns
    -------
    pair_distances : list
        A list of all distances between pairs of atoms in the image.

    Examples
    --------
    s = am.dummy_data.get_simple_cubic_signal()
    sublattice = am.dummy_data.get_simple_cubic_sublattice()
    pdf = pair_distribution_function(s,sublattice.atom_positions)

    """
    pair_distances = []
    distance_from_edge = []

    x_size = image.axes_manager[0].size
    y_size = image.axes_manager[1].size
    x_scale = image.axes_manager[0].scale
    y_scale = image.axes_manager[1].scale
    units = image.axes_manager[0].units

    for position1 in atom_positions:
        distance_from_edge.append(
                [min([position1[0]*x_scale, (x_size-position1[0])*x_scale]),
                 min([position1[1]*y_scale, (y_size-position1[1])*y_scale])])
        for position2 in atom_positions:
            if not np.array_equal(position1, position2):
                pair_distance = ((position1[0] * x_scale-position2[0] *
                                  x_scale)**2+(position1[1] * y_scale -
                                  position2[1] * y_scale)**2)**0.5
                pair_distances.append(pair_distance)

    if plot:
        plt.figure()
        intensities, bins, patches = plt.hist(
                pair_distances, bins=n_bins, density=True, histtype='step',
                range=(0, rel_range * min([x_size, y_size]) *
                       min([x_scale, y_scale])/2))
        plt.clf()
        for i, intensity in enumerate(intensities):
            intensity = intensity/(1+2*i)
            area_correction = []
            for distance in distance_from_edge:
                if min(distance) < bins[i+1]:
                    area_correction.append(
                            1-_area_proportion(distance, bins[i+1]))
                else:
                    area_correction.append(1)
            if len(area_correction) > 0:
                mean = sum(area_correction)/len(area_correction)
            else:
                mean = 1
            intensities[i] = intensity/mean
        plt.plot(bins[1:], intensities)
        if isinstance(units, str):
            plt.xlabel('Distance (' + units + ')')
        else:
            plt.xlabel('Distance (pixels)')
        plt.show()

    return pair_distances


def _area_proportion(distances, radius):
    distances = -(distances-radius)
    dist_norm = [(i > 0) * i for i in distances]
    if 0 in dist_norm:
        proportion = math.acos(max(dist_norm)/radius)/math.pi
    else:
        proportion = 0.25 + (math.acos(dist_norm[0]/radius) +
                             math.acos(dist_norm[1]/radius))/(2*math.pi)
    return proportion
