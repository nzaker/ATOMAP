import math
import numpy as np
from numpy.random import normal
from hyperspy.signals import Signal2D
from hyperspy.misc.utils import isiterable
from atomap.external.gaussian2d import Gaussian2D
from atomap.sublattice import Sublattice
from atomap.atom_position import Atom_Position


def get_simple_cubic_signal():
    simple_cubic = TestData(300, 300)
    x, y = np.mgrid[10:290:15j, 10:290:15j]
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3)
    return simple_cubic.signal

def get_simple_cubic_sublattice():
    simple_cubic = TestData(300, 300)
    x, y = np.mgrid[10:290:15j, 10:290:15j]
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3)
    sublattice = simple_cubic.sublattice
    sublattice.image = simple_cubic.signal.data
    sublattice.original_image = simple_cubic.signal.data
    return sublattice


class TestData(object):

    def __init__(self, image_x, image_y):
        self.data_extent = (image_x, image_y)
        self.sublattice = Sublattice([], None)
        self.sublattice.atom_list = []

    @property
    def signal(self):
        signal = self.sublattice.get_model_image(
                image_shape=self.data_extent, progressbar=False)
        return signal

    @property
    def gaussian_list(self):
        gaussian_list = []
        for atom in self.sublattice.atom_list:
            gaussian_list.append(atom.as_gaussian())
        return gaussian_list

    def add_atom(self, x, y, sigma_x=1, sigma_y=1, amplitude=1, rotation=0):
        atom = Atom_Position(
                x=x, y=y, sigma_x=sigma_x, sigma_y=sigma_y,
                rotation=rotation, amplitude=amplitude)
        self.sublattice.atom_list.append(atom)

    def add_atom_list(
            self, x, y, sigma_x=1, sigma_y=1, amplitude=1, rotation=0):
        """
        Add several atoms.

        x, y : iterable
            Position of the atoms. Must be iterable, and have the same size.
        sigma_x, sigma_y : number or iterable, default 1
            If number: all the atoms will have the same sigma.
            Use iterable for setting different sigmas for different atoms.
            If iterable: must be same length as x and y iterables.
        amplitude : number or iterable, default 1
            If number: all the atoms will have the same amplitude.
            Use iterable for setting different amplitude for different atoms.
            If iterable: must be same length as x and y iterables.
        rotation : number or iterable, default 1
            If number: all the atoms will have the same rotation.
            Use iterable for setting different rotation for different atoms.
            If iterable: must be same length as x and y iterables.
        """
        if len(x) != len(y):
            raise ValueError("x and y needs to have the same length")

        if isiterable(sigma_x):
            if len(sigma_x) != len(x):
                raise ValueError("sigma_x and x needs to have the same length")
        else:
            sigma_x = [sigma_x]*len(x)

        if isiterable(sigma_y):
            if len(sigma_y) != len(y):
                raise ValueError("sigma_y and x needs to have the same length")
        else:
            sigma_y = [sigma_y]*len(x)

        if isiterable(amplitude):
            if len(amplitude) != len(x):
                raise ValueError(
                        "amplitude and x needs to have the same length")
        else:
            amplitude = [amplitude]*len(x)

        if isiterable(rotation):
            if len(rotation) != len(x):
                raise ValueError(
                        "rotation and x needs to have the same length")
        else:
            rotation = [rotation]*len(x)
        iterator = zip(x, y, sigma_x, sigma_y, amplitude, rotation)
        for tx, ty, tsigma_x, tsigma_y, tamplitude, trotation in iterator:
            self.add_atom(tx, ty, tsigma_x, tsigma_y, tamplitude, trotation)


def get_test_dumbbell_signal():
    x_list, y_list = [], []
    for x in range(10, 200, 20):
        for y in range(10, 200, 20):
            x_list.append(x)
            y_list.append(y)
    for x in range(16, 200, 20):
        for y in range(10, 200, 20):
            x_list.append(x)
            y_list.append(y)
    s, g_list = make_artifical_atomic_signal(x_list, y_list, image_pad=0)
    return(s)


def make_artifical_atomic_signal(
        x, y, sigma_x=None, sigma_y=None, A=None, rotation=None,
        image_pad=50.):
    """
    Make an atomic resolution artificial image, by modeling the
    atomic columns as Gaussian distributions.

    Parameters
    ----------
    x, y : 1D list
        x and y positions. The image_pad value will be added
        to each position, meaning that if a position is 
        x=[10], y=[20], and image_pad=15. The position on the image
        will be x=25, y=35
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
        int((max_size_y - min_size_y + image_pad*2 + 1)),
        int((max_size_x - min_size_x + image_pad*2 + 1)))))
    model = temp_signal.create_model()
    model.extend(gaussian_list)
    signal = model.as_signal()
    signal.metadata.General.title = "artificial_signal"
    return signal, gaussian_list


def make_vector_test_gaussian(x, y, standard_deviation=1, n=30):
    point_list = []
    for i in range(n):
        g_x = normal(x, scale=standard_deviation)
        g_y = normal(y, scale=standard_deviation)
        point_list.append([g_x, g_y])
    point_list = np.array(point_list)
    return(point_list)


def make_nn_test_dataset(xN=3, yN=3, xS=9, yS=9, std=0.3, n=50):
    point_list = np.array([[],[]]).T
    for ix in range(-xN, xN+1):
        for iy in range(-yN, yN+1):
            if (ix == 0) and (iy == 0):
                pass
            else:
                gaussian_list = make_vector_test_gaussian(
                        ix*xS, iy*yS, standard_deviation=std, n=n)
                point_list = np.vstack((point_list, gaussian_list))
    return(point_list)


def find_atom_position_match(component_list, atom_list, delta=3, scale=1.):
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
