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
    Fantasite is a fantastic structure with several interesting structural
    variations. It contains two sublattices, doimains with elliptical atomic
    columns and tilt-patterns.
    """
    test_data = MakeTestData(500, 500)
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
    for i in range(3*7*17+17,580,17):
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
        
    xB0, yB0 = np.mgrid[10:495:15, 25:495:30]
    xB0, yB0 = xB0.flatten(), yB0.flatten()
    test_data.add_atom_list(xB0[0:8*16],yB0[0:8*16],
                sigma_x=3, sigma_y=3, amplitude=20)
    xB2, yB2 = xB0[8*16:], yB0[8*16:]
    sig = np.arange(3,4.1,0.2)
    sigma_y_list = np.hstack((sig, sig[::-1], sig, sig[::-1],np.full(10,3)))
    down = True
    for i, x in enumerate(xB2):
        rotation = 0.39
        if down:
            rotation *= -1
        sigma_y = sigma_y_list[i // 16]
        test_data.add_atom(x, yB2[i], sigma_x=3, sigma_y=sigma_y, 
                           amplitude=20, rotation=rotation)
        down = not down
    test_data.add_image_noise(mu=0, sigma=0.01)
    return test_data.signal
