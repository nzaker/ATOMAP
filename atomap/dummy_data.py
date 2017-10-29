import numpy as np
import hyperspy.api as hs
from atomap.testing_tools import MakeTestData
from atomap.atom_lattice import Atom_Lattice


def _make_simple_cubic_testdata(image_noise=False):
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:20j, 10:290:20j]
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3)
    if image_noise:
        simple_cubic.add_image_noise(mu=0, sigma=0.002)
    return simple_cubic


def get_simple_cubic_signal(image_noise=False):
    """Generate a test image signal of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> s.plot()

    """
    test_data = _make_simple_cubic_testdata(image_noise=image_noise)
    return test_data.signal


def get_simple_cubic_sublattice(image_noise=False):
    """Generate a test sublattice of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    sublattice : Atomap Sublattice

    Examples
    --------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.plot()

    """
    test_data = _make_simple_cubic_testdata(image_noise=image_noise)
    return test_data.sublattice


def _make_atom_lists_two_sublattices(test_data_1, test_data_2=None):
    x0, y0 = np.mgrid[10:295:20, 10:300:34]
    test_data_1.add_atom_list(
            x0.flatten(), y0.flatten(), sigma_x=3, sigma_y=3, amplitude=20)

    if test_data_2 is None:
        test_data_2 = test_data_1

    x1, y1 = np.mgrid[10:295:20, 27:290:34]
    test_data_2.add_atom_list(
            x1.flatten(), y1.flatten(), sigma_x=3, sigma_y=3, amplitude=10)


def get_two_sublattice_signal():
    test_data = MakeTestData(300, 300)
    x0, y0 = np.mgrid[10:295:20, 10:300:34]
    test_data.add_atom_list(
            x0.flatten(), y0.flatten(), sigma_x=3, sigma_y=3, amplitude=20)

    x1, y1 = np.mgrid[10:295:20, 27:290:34]
    test_data.add_atom_list(
            x1.flatten(), y1.flatten(), sigma_x=3, sigma_y=3, amplitude=10)

    test_data.add_image_noise(mu=0, sigma=0.01)
    return test_data.signal


def get_simple_atom_lattice_two_sublattices(image_noise=False):
    """Returns a simple atom_lattice with two sublattices.

    Parameters
    ----------
    image_noise : bool, default False

    Returns
    -------
    simple_atom_lattice : Atom_Lattice object

    Examples
    --------
    >>> import atomap.api as am
    >>> al = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> al.plot()

    """
    test_data_1 = MakeTestData(300, 300)
    test_data_2 = MakeTestData(300, 300)
    test_data_3 = MakeTestData(300, 300)
    if image_noise:
        test_data_3.add_image_noise(mu=0, sigma=0.02)
    _make_atom_lists_two_sublattices(test_data_1, test_data_2)
    _make_atom_lists_two_sublattices(test_data_3)

    sublattice_1 = test_data_1.sublattice
    sublattice_2 = test_data_2.sublattice
    sublattice_2._plot_color = 'b'
    image = test_data_3.signal.data
    atom_lattice = Atom_Lattice(
            image=image, name='Simple Atom Lattice',
            sublattice_list=[sublattice_1, sublattice_2])
    return(atom_lattice)


def get_simple_heterostructure_signal(image_noise=True):
    test_data = MakeTestData(400, 400)
    x0, y0 = np.mgrid[10:390:15, 10:200:15]
    test_data.add_atom_list(
            x0.flatten(), y0.flatten(), sigma_x=3, sigma_y=3, amplitude=5)

    y0_max = y0.max()
    x1, y1 = np.mgrid[10:390:15, y0_max+16:395:16]
    test_data.add_atom_list(
            x1.flatten(), y1.flatten(), sigma_x=3, sigma_y=3, amplitude=5)

    if image_noise:
        test_data.add_image_noise(mu=0.0, sigma=0.005)
    return test_data.signal


def get_dumbbell_signal():
    test_data = MakeTestData(200, 200)
    x0, y0 = np.mgrid[10:200:20, 10:200:20]
    x1, y1 = np.mgrid[10:200:20, 16:200:20]
    x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
    test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
    return test_data.signal


def _add_fantasite_sublattice_A(test_data):
    xA0, yA0 = np.mgrid[10:495:15, 10:495:30]
    xA0, yA0 = xA0.flatten(), yA0.flatten()
    xA1, yA1 = xA0[0:8*17], yA0[0:8*17]
    test_data.add_atom_list(xA1, yA1, sigma_x=3, sigma_y=3, amplitude=10)
    dx = 1
    for i in range(8*17, 3*7*17, 2*17):
        xA2 = xA0[i:i+17] + dx
        xA3 = xA0[i+17:i+34] - dx
        yA2, yA3 = yA0[i:i+17], yA0[i+17:i+34]
        test_data.add_atom_list(xA2, yA2, sigma_x=3, sigma_y=3, amplitude=10)
        test_data.add_atom_list(xA3, yA3, sigma_x=3, sigma_y=3, amplitude=10)
    down = True
    for i in range(3*7*17+17, 580, 17):
        xA4, xA5 = xA0[i:i+17:2], xA0[i+1:i+17:2]
        if down:
            yA4 = yA0[i:i+17:2] + dx
            yA5 = yA0[i+1:i+17:2] - dx
        if not down:
            yA4 = yA0[i:i+17:2] - dx
            yA5 = yA0[i+1:i+17:2] + dx
        test_data.add_atom_list(xA4, yA4, sigma_x=3, sigma_y=3, amplitude=10)
        test_data.add_atom_list(xA5, yA5, sigma_x=3, sigma_y=3, amplitude=10)
        down = not down
    test_data.add_image_noise(mu=0, sigma=0.01, random_seed=10)
    return test_data


def _add_fantasite_sublattice_B(test_data):
    xB0, yB0 = np.mgrid[10:495:15, 25:495:30]
    xB0, yB0 = xB0.flatten(), yB0.flatten()
    test_data.add_atom_list(
            xB0[0:8*16], yB0[0:8*16],
            sigma_x=3, sigma_y=3, amplitude=20)
    xB2, yB2 = xB0[8*16:], yB0[8*16:]
    sig = np.arange(3, 4.1, 0.2)
    sigma_y_list = np.hstack((sig, sig[::-1], sig, sig[::-1], np.full(10, 3)))
    down = True
    for i, x in enumerate(xB2):
        rotation = 0.39
        if down:
            rotation *= -1
        sigma_y = sigma_y_list[i // 16]
        test_data.add_atom(
                x, yB2[i], sigma_x=3, sigma_y=sigma_y,
                amplitude=20, rotation=rotation)
        down = not down
    test_data.add_image_noise(mu=0, sigma=0.01, random_seed=0)
    return test_data


def _get_fantasite_atom_lattice():
    test_data1 = MakeTestData(500, 500)
    test_data1 = _add_fantasite_sublattice_A(test_data1)
    test_data2 = MakeTestData(500, 500)
    test_data2 = _add_fantasite_sublattice_B(test_data2)
    test_data3 = MakeTestData(500, 500)
    test_data3 = _add_fantasite_sublattice_A(test_data3)
    test_data3 = _add_fantasite_sublattice_B(test_data3)
    test_data3.add_image_noise(mu=0, sigma=0.01, random_seed=0)

    sublattice_1 = test_data1.sublattice
    sublattice_2 = test_data2.sublattice
    sublattice_1._plot_color = 'b'
    image = test_data3.signal.data
    atom_lattice = Atom_Lattice(
            image=image, name='Fantasite Atom Lattice',
            sublattice_list=[sublattice_1, sublattice_2])
    return(atom_lattice)


def _make_fantasite_test_data():
    test_data = MakeTestData(500, 500)
    test_data = _add_fantasite_sublattice_A(test_data)
    test_data = _add_fantasite_sublattice_B(test_data)
    test_data.add_image_noise(mu=0, sigma=0.01, random_seed=0)
    return test_data


def get_fantasite():
    """
    Fantasite is a fantastic structure with several interesting structural
    variations.

    It contains two sublattices, domains with elliptical atomic
    columns and tilt-patterns. This function returns a HyperSpy 2D signal.

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_fantasite()
    >>> s.plot()

    Returns
    -------
    fantasite_signal : HyperSpy 2D signal

    See also
    --------
    get_fantasite_sublattice : get a sublattice object of the fantasite.

    """
    test_data = _make_fantasite_test_data()
    return test_data.signal


def get_fantasite_sublattice():
    """
    Fantasite is a fantastic structure with several interesting structural
    variations.

    It contains two sublattices, domains with elliptical atomic
    columns and tilt-patterns. This function returns an Atomap sublattice.

    Currently this function returns the two sublattices as one sublattice.
    To get these sublattices separately, see get_fantasite_atom_lattice

    Examples
    --------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_fantasite_sublattice()
    >>> sublattice.plot()

    See also
    --------
    get_fantasite : get a Signal2D object of the fantasite.
    get_fantasite_atom_lattice : get the atom lattice
        for fantasite, with both sublattices.

    """
    test_data = _make_fantasite_test_data()
    return test_data.sublattice


def get_fantasite_atom_lattice():
    """
    Fantasite is a fantastic structure with several interesting structural
    variations.

    It contains two sublattices, domains with elliptical atomic
    columns and tilt-patterns. This function returns an Atomap Atom_Lattice.

    Examples
    --------
    >>> import atomap.api as am
    >>> atom_lattice = am.dummy_data.get_fantasite_atom_lattice()
    >>> atom_lattice.plot()

    """
    atom_lattice = _get_fantasite_atom_lattice()
    return(atom_lattice)


def get_perovskite110_ABF_signal(image_noise=False):
    """Returns signals representing an ABF image of a perovskite <110>.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D,

    Examples
    --------
    >>> import atomap.api as am
    >>> s_ABF = am.dummy_data.get_perovskite110_ABF_signal()

    """
    test_data = MakeTestData(300, 300)
    x_A, y_A = np.mgrid[10:295:20, 10:300:34]
    test_data.add_atom_list(x_A.flatten(), y_A.flatten(),
                            sigma_x=5, sigma_y=5, amplitude=40)
    x_B, y_B = np.mgrid[10:295:20, 27:290:34]
    test_data.add_atom_list(x_B.flatten(), y_B.flatten(),
                            sigma_x=3, sigma_y=3, amplitude=10)
    x_O, y_O = np.mgrid[0:295:20, 27:290:34]
    test_data.add_atom_list(x_O.flatten(), y_O.flatten(),
                            sigma_x=2, sigma_y=2, amplitude=1.2)

    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.01)

    ABF = 1-test_data.signal.data
    s_ABF = hs.signals.Signal2D(ABF)
    return(s_ABF)
