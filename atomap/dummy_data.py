import numpy as np
from atomap.testing_tools import MakeTestData


def get_simple_cubic_signal(image_noise=False):
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:20j, 10:290:20j]
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3)
    if image_noise:
        simple_cubic.add_image_noise(mu=0, sigma=0.002)
    return simple_cubic.signal


def get_simple_cubic_sublattice():
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:20j, 10:290:20j]
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


def get_simple_heterostructure_signal():
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


def get_dumbbell_signal():
    test_data = MakeTestData(200, 200)
    x0, y0 = np.mgrid[10:200:20, 10:200:20]
    x1, y1 = np.mgrid[10:200:20, 16:200:20]
    x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
    test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
    return test_data.signal
    
def get_fantasite():
    """
    Fantasite is a fantasy structure in which several interesting structural
    variations occur. It contains two sublattices.
    """
    test_data = MakeTestData(300, 300)
    xA0, yA0 = np.mgrid[10:295:20, 10:300:34]
    xA0, yA0 = xA0.flatten(), yA0.flatten()
    xA1, yA1 = xA0[0:(9*5)], yA0[0:(9*5)]
    test_data.add_atom_list(xA1, yA1, sigma_x=3, sigma_y=3, amplitude=10)
    dx = 1
    for i in range(45, 90, 18):
        xA2 = xA0[i:i+9] + dx
        xA3 = xA0[i+9:i+18] - dx
        yA2, yA3 = yA0[i:i+9], yA0[i+9:i+18]
        test_data.add_atom_list(xA2, yA2, sigma_x=3, sigma_y=3, amplitude=10)
        test_data.add_atom_list(xA3, yA3, sigma_x=3, sigma_y=3, amplitude=10)
    for i in range(99,135,18):
        xA4, xA5 = xA0[i:i+9], xA0[i+9:i+18]
        yA4 = yA0[i:i+9] + dx
        yA5 = yA0[i+9:i+18] - dx
        test_data.add_atom_list(xA4, yA4, sigma_x=3, sigma_y=3, amplitude=10)
        test_data.add_atom_list(xA5, yA5, sigma_x=3, sigma_y=3, amplitude=10)

    xB1, yB1 = np.mgrid[10:295:20, 27:290:34]
    test_data.add_atom_list(
                xB1.flatten()[:40], yB1.flatten()[:40],
                sigma_x=3, sigma_y=3, amplitude=20)
    xB2, yB2 = xB1.flatten()[40:], yB1.flatten()[40:]
    sigma_y_list = [3, 3.2, 3.4, 3.6, 3.8, 4, 3.8, 3.6, 3.4, 3.2, 3]
    for i, x in enumerate(xB2):
        sigma_y = sigma_y_list[i // 8]
        test_data.add_atom(x, yB2[i], sigma_x=3, sigma_y=sigma_y, 
                           amplitude=20, rotation=0.39)
    test_data.add_image_noise(mu=0, sigma=0.01)
    return test_data.signal
