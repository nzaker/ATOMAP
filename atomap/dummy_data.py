import numpy as np
from atomap.testing_tools import MakeTestData


def get_simple_cubic_signal():
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:15j, 10:290:15j]
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3)
    return simple_cubic.signal


def get_simple_cubic_sublattice():
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:15j, 10:290:15j]
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3)
    sublattice = simple_cubic.sublattice
    sublattice.image = simple_cubic.signal.data
    sublattice.original_image = simple_cubic.signal.data
    return sublattice


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


def get_simple_heterostructure():
    test_data = MakeTestData(400, 400)
    x0, y0 = np.mgrid[10:390:15, 10:200:15]
    test_data.add_atom_list(
            x0.flatten(), y0.flatten(), sigma_x=3, sigma_y=3, amplitude=5)

    y0_max = y0.max()
    x1, y1 = np.mgrid[10:390:15, y0_max+16:395:16]
    test_data.add_atom_list(
            x1.flatten(), y1.flatten(), sigma_x=3, sigma_y=3, amplitude=5)

    test_data.add_image_noise(mu=0.0, sigma=0.005)
    return test_data.signal
