from hyperspy.signals import Signal2D
from atomap.external.gaussian2d import Gaussian2D
import numpy as np
import math


def make_artifical_atomic_signal(
        x, y, sigma_x=None, sigma_y=None, A=None, rotation=None,
        image_pad=50.):
    """
    Make an atomic resolution artificial image, by modeling the
    atomic columns as Gaussian distributions.

    Parameters
    ----------
    x : 1D list
    y : 1D list
        x and y positions
    sigma_x : 1D list, optional, default 1
    sigma_y : 1D list, optional, default 1
    A : 1D list, optional, default 1
        Amplitude of the Gaussian
    rotation : 1D list, optional, default 1
    image_pad : optional, default 50

    Returns
    -------
    Signal2D and list of components used to generate the signal.

    Examples
    --------
    >>> import numpy as np
    >>> from atomap.testing_tools import make_artifical_atomic_signal
    >>> x, y = np.mgrid[0:50:5j,0:50:5j]
    >>> x, y = x.flatten(), y.flatten()
    >>> s, g_list = make_artifical_atomic_signal(x, y, image_pad=10)
    >>> s.plot()
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if sigma_x is None:
        sigma_x = [1.]*len(x)
    if sigma_y is None:
        sigma_y = [1.]*len(x)
    if A is None:
        A = [1.]*len(x)
    if rotation is None:
        rotation = [0.]*len(x)
    iterator = zip(x, y, sigma_x, sigma_y, A, rotation)
    gaussian_list = []
    for tx, ty, tsigma_x, tsigma_y, tA, trotation in iterator:
        g = Gaussian2D(
                A=tA,
                sigma_x=tsigma_x,
                sigma_y=tsigma_y,
                centre_x=tx+image_pad,
                centre_y=ty+image_pad,
                rotation=trotation)
        gaussian_list.append(g)
    min_size_x, max_size_x = min(x), max(x)
    min_size_y, max_size_y = min(y), max(y)
    temp_signal = Signal2D(np.zeros((
        int((max_size_x-min_size_x+image_pad*2)),
        int((max_size_y-min_size_y+image_pad*2)))))
    model = temp_signal.create_model()
    model.extend(gaussian_list)
    signal = model.as_signal()
    signal.metadata.General.title = "artificial_signal"
    return signal, gaussian_list


def find_atom_position_match(component_list, atom_list, delta=3, scale=1.):
    delta = 10
    match_list = []
    for atom in atom_list:
        for component in component_list:
            x = atom.pixel_x*scale - component.centre_x.value
            y = atom.pixel_y*scale - component.centre_y.value
            d = math.hypot(x, y)
            if d < delta:
                match_list.append([component, atom])
                break
    return match_list


def get_fit_miss_array(match_list):
    fit_miss = []
    for match in match_list:
        x = match[0].centre_x.value - match[1].pixel_x
        y = match[0].centre_y.value - match[1].pixel_y
        d = math.hypot(x, y)
        fit_miss.append([x, y, d])
    fit_miss = np.array(fit_miss)
    return fit_miss
