import numpy as np
from tqdm import tqdm
from atomap.atom_finding_refining import get_atom_positions
from atomap.tools import _get_n_nearest_neighbors, Fingerprinter
from atomap.atom_finding_refining import _make_circular_mask, do_pca_on_signal
from atomap.sublattice import Sublattice
from atomap.atom_lattice import Dumbbell_Lattice
from operator import itemgetter


def find_dumbbell_vector(s, separation):
    """
    Find the vector between the atoms in structure with dumbbells.
    For example GaAs-(110).

    Parameters
    ----------
    s : HyperSpy 2D signal
    separation : int
        Pixel separation between the atoms in the dumbbells.

    Returns
    -------
    dumbbell_vector : tuple

    Examples
    --------
    >>> import atomap.api as am
    >>> from atomap.initial_position_finding import find_dumbbell_vector
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> dumbbell_vector = find_dumbbell_vector(s, 4)
    """
    position_list = get_atom_positions(s, separation=separation)
    test = _get_n_nearest_neighbors(position_list, 10, leafsize=100)
    fp = Fingerprinter()
    fp.fit(test)
    clusters = fp.cluster_centers_
    clusters_dist = (clusters[:, 0]**2+clusters[:, 1]**2)**0.5
    sort_zip = zip(list(clusters_dist), clusters[:, 0], clusters[:, 1])
    cluster_dist, cluster_x, cluster_y = zip(
            *sorted(sort_zip, key=itemgetter(0)))
    vec0 = cluster_x[0], cluster_y[0]
    vec1 = cluster_x[1], cluster_y[1]

    if (abs(vec0[0]+vec1[0]) > 0.1) or (abs(vec0[1]+vec1[1]) > 0.1):
        raise ValueError(
                "Dumbbell vectors should be antiparallel, but are %r and %r"
                % (vec0, vec1))
    return(vec0)


def _get_dumbbell_arrays(
        s, position_list, dumbbell_vector, show_progressbar=True):
    """
    Parameters
    ----------
    s : HyperSpy 2D signal
    position_list : list of atomic positions
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    dumbbell_vector : tuple
    show_progressbar : bool, default True

    Returns
    -------
    Dumbbell lists : tuple of lists

    Examples
    --------
    >>> import atomap.api as am
    >>> from atomap.initial_position_finding import find_dumbbell_vector
    >>> from atomap.initial_position_finding import _get_dumbbell_arrays
    >>> from atomap.atom_finding_refining import get_atom_positions
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> position_list = get_atom_positions(s, separation=16)
    >>> dumbbell_vector = find_dumbbell_vector(s, 4)
    >>> d0, d1 = _get_dumbbell_arrays(s, position_list, dumbbell_vector)
    """
    next_pos_list0 = []
    next_pos_list1 = []
    for x, y in zip(position_list[:, 0], position_list[:, 1]):
        next_pos_list0.append([dumbbell_vector[0]+x, dumbbell_vector[1]+y])
        next_pos_list1.append([-dumbbell_vector[0]+x, -dumbbell_vector[1]+y])
    next_pos_list0 = np.array(next_pos_list0)
    next_pos_list1 = np.array(next_pos_list1)

    mask_radius = 0.5*(dumbbell_vector[0]**2+dumbbell_vector[1]**2)**0.5

    iterator = zip(
            position_list[:, 0], position_list[:, 1],
            next_pos_list0, next_pos_list1)
    total_num = len(next_pos_list0)
    dumbbell_list0, dumbbell_list1 = [], []
    for x, y, next_pos0, next_pos1 in tqdm(
            iterator, total=total_num, desc="Finding dumbbells",
            disable=not show_progressbar):
        mask1 = _make_circular_mask(
                next_pos0[1], next_pos0[0],
                s.data.shape[0], s.data.shape[1],
                mask_radius)
        mask2 = _make_circular_mask(
                next_pos1[1], next_pos1[0],
                s.data.shape[0], s.data.shape[1],
                mask_radius)
        pos1_sum = (s.data*mask1).sum()
        pos2_sum = (s.data*mask2).sum()
        if pos1_sum > pos2_sum:
            dumbbell_list0.append([x, y])
            dumbbell_list1.append(next_pos0)
        else:
            dumbbell_list0.append(next_pos1)
            dumbbell_list1.append([x, y])
    dumbbell_list0 = np.array(dumbbell_list0)
    dumbbell_list1 = np.array(dumbbell_list1)
    return(dumbbell_list0, dumbbell_list1)


def make_atom_lattice_dumbbell_structure(
        s, position_list, dumbbell_vector, show_progressbar=True):
    """
    Make Atom_Lattice object from image of dumbbell structure.

    Parameters
    ----------
    s : HyperSpy 2D signal
    position_list : list of atomic positions
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    dumbbell_vector : tuple
    show_progressbar : bool, default True

    Returns
    -------
    dumbbell_lattice: Atomap Dumbbell_Lattice object

    Examples
    --------
    >>> import atomap.api as am
    >>> import atomap.initial_position_finding as ipf
    >>> from atomap.atom_finding_refining import get_atom_positions
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> position_list = get_atom_positions(s, separation=16)
    >>> dumbbell_vector = ipf.find_dumbbell_vector(s, 4)
    >>> dumbbell_lattice = ipf.make_atom_lattice_dumbbell_structure(
    ...     s, position_list, dumbbell_vector)
    """
    dumbbell_list0, dumbbell_list1 = _get_dumbbell_arrays(
            s, position_list, dumbbell_vector,
            show_progressbar=show_progressbar)
    s_modified = do_pca_on_signal(s)
    sublattice0 = Sublattice(
            atom_position_list=dumbbell_list0,
            original_image=s.data,
            image=s_modified.data,
            color='blue')
    sublattice1 = Sublattice(
            atom_position_list=dumbbell_list1,
            original_image=s.data,
            image=s_modified.data,
            color='red')
    sublattice0.find_nearest_neighbors()
    sublattice1.find_nearest_neighbors()
    atom_lattice = Dumbbell_Lattice(
            image=sublattice0.image,
            name="Dumbbell structure",
            sublattice_list=[sublattice0, sublattice1])
    return(atom_lattice)
