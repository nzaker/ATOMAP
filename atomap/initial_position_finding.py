from atomap.atom_finding_refining import get_atom_positions
from atomap.tools import _get_n_nearest_neighbors, Fingerprinter
import operator

def _find_dumbbell_vector(s, separation):
    position_list = get_atom_positions(s, separation=separation)
    test = _get_n_nearest_neighbors(position_list, 10, leafsize=100)
    fp = Fingerprinter()
    fp.fit(test)
    clusters = fp.cluster_centers_

    clusters_distance = (clusters[:,0]**2+clusters[:,1]**2)**0.5

    sort_zip = zip(list(clusters_distance), clusters[:,0], clusters[:,1])

    cluster_distance, cluster_x, cluster_y = zip(*sorted(sort_zip, key=operator.itemgetter(0)))

    vec0 = cluster_x[0], cluster_y[0]
    vec1 = cluster_x[1], cluster_y[1]

    if (abs(vec0[0]+vec1[0]) > 0.1) or (abs(vec0[1]+vec1[1]) > 0.1):
        raise ValueError(
                "Dumbbell vectors should be antiparallel, but are %r and %r"
                % (vec0, vec1))


    return(vec0)

