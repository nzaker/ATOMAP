import numpy as np
import scipy
import matplotlib.pyplot as plt


def centeredDistanceMatrix(centre, det_image):
    """Makes a matrix the same size as det_image centre around a particular point.

    Parameters
    ----------
    centre : tuple
            x and y position of where the centre of the matrix should be.
    det_image : NumPy array
            Detector map as 2D array.

    Returns
    -------
    NumPy Array

    """
    # makes a matrix centre around tuble 'centre' the same size as the det_image.
    x, y = np.meshgrid(range(det_image.shape[0]),
                       range(det_image.shape[0]))
    return np.sqrt((x - (centre[1]) + 1)**2 + (y - (centre[0]))**2)


def _detector_threshold(det_image):
    """Using an input detector image. The function thresholds returning a
    binary image.

    Parameters
    ----------
    det_image : NumPy array
            Detector map as 2D array.

    Returns
    -------
    threshold_image : NumPy array
            Boolean image demonstrating the active region of the detector.

    """

    threshold_image = np.zeros_like(det_image)
    threshold_image = (det_image - np.min(det_image)) / \
        (np.max(det_image) - np.min(det_image))
    threshold_image = (threshold_image >= 0.25)
    return threshold_image


def _func(x, a, b, c):
    return a * (x**-b) + c


def _radial_profile(data, centre):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


class InteractiveFluxAnalyser:

    def __init__(self, profile, radius, flux_profile):
        self.profile = profile[0]
        self.radius = radius
        self.flux_profile = flux_profile
        self.left = np.int(min(self.profile.get_xdata()))
        self.right = np.int(max(self.profile.get_xdata()))
        self.l_line = self.profile.axes.axvline(self.left, color='firebrick',
                                                linestyle='--')
        self.r_line = self.profile.axes.axvline(self.right, color='seagreen',
                                                linestyle='--')
        self.cid = self.profile.figure.canvas.mpl_connect('button_press_event',
                                                          self.onclick)

    def __call__(self):

        print('Final coordinates are: {}, {}!'.format(self.left, self.right))
        self.coords = [self.left, self.right]

    def onclick(self, event):
        # print('click', vars(event))
        if event.inaxes != self.profile.axes:
            return
        if event.button == 1:  # Left mouse button
            left = np.int(event.xdata)
            if left < self.right:
                self.left = left
            self.l_line.set_xdata(self.left)
            self.profile.figure.canvas.draw_idle()
        elif event.button == 3:  # Right mouse button
            right = np.int(event.xdata)
            if right > self.left:
                self.right = right
            self.r_line.set_xdata(self.right)
            self.profile.figure.canvas.draw_idle()
        elif event.button == 2:  # Middle mouse button
            event.canvas.mpl_disconnect(self.cid)
            self()
        print('Coordinates selected', self.left, self.right)


def find_flux_limits(flux_pattern, conv_angle):
    """Using an input flux_pattern to create a line profile. The user is then
    able to select the region of this profile which follows an exponential.

    Parameters
    ----------

    flux_pattern : NumPy array

    conv_angle : float

    Returns
    -------
    profiler:

    """
    # normalise flux image to be scaled 0-1.
    low_values_indices = flux_pattern < 0
    flux_pattern[low_values_indices] = 0
    flux_pattern = flux_pattern / np.max(flux_pattern)

    # convert 2D image into a 1D profile.
    centre = scipy.ndimage.measurements.center_of_mass(flux_pattern)
    flux_profile = _radial_profile(flux_pattern, centre[::-1])
    grad = np.gradient(flux_profile)
    # scale flux_profile relative to the bright field disc.
    x = np.array(range(flux_profile.shape[0]))
    radius = x * conv_angle / np.argmin(grad)

    # Plot the radial flux profile and allow the user to select the region for
    # power-law fitting.
    fig = plt.figure()
    plt.suptitle('Radial Flux Profile: Please select power-law region.',
                 fontsize=10)
    ax1 = fig.add_subplot(2, 1, 1)
    plt.plot(radius, flux_profile)
    ax1.set_title('Radial Profile')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('Logarithmic Profile')
    profile = plt.plot(radius, flux_profile)
    ax2.set_yscale('log')
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    profiler = InteractiveFluxAnalyser(profile, radius, flux_profile)

    return(profiler, flux_profile)


def analyse_flux(coords, flux_profile, conv_angle):
    """Using an input detector image and flux image, a detector normalisation
    is applied with regards to

    Parameters
    ----------
    image : NumPy array
            Experimental image to be normalised.
    det_image : NumPy array
            Detector map as 2D array.
    flux_image : 'None' otherwise NumPy array
            Flux image is required for a flux weighted detector normalisation,
            if none is supplied the normal thresholding normalisation will
            be applied.

    Returns
    -------
    normalised_image : NumPy array
            The experimental image after detector normalisation such that,
            all intensities are a fraction of the incident beam.
    """
    grad = np.gradient(flux_profile)
    x = np.array(range(flux_profile.shape[0]))
    radius = x * conv_angle / np.argmin(grad)
    lower = np.sum(radius < coords[0]) - 1
    upper = np.sum(radius < coords[1])
    xdata = radius[lower:upper]
    ydata = flux_profile[lower:upper]

    popt, pcov = scipy.optimize.curve_fit(_func, xdata, ydata, p0=([1, 1, 1]))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata, _func(xdata, *popt), 'r-', label='fit')
    ax1.set_yscale('log')
    plt.legend()
    plt.suptitle('Resulting fitted profile, fontsize=10')

    outer_cutoff = radius[np.argmin(grad[upper:]) + upper]

    return(popt[1], outer_cutoff)


def detector_normalisation(image,
                           det_image,
                           inner_angle,
                           outer_angle='None',
                           flux_expo='None'):
    """Using an input detector image and flux exponent (if provided) a detector
    normalisation is carried out on the experimental image.

    Parameters
    ----------
    image : NumPy array
            Experimental image to be normalised.
    det_image : NumPy array
            Detector map as 2D array.
    inner_angle: float
            The experimentally measured inner collection angle of the detector.
    outer_angle: 'None' otherwise float
            The measured experimental detector collection angle. If left as
            'None' the outer limit of the detector active region will be used.
    flux_expo: 'None' otherwise float
            flux_expo is required to carry out flux_weighted detector
            normalisation procedure. For more details see:
            Martinez et al. Ultramicroscopy 2015, 159, 46–58.

    Returns
    -------
    normalised_image : NumPy array
            The experimental image after detector normalisation such that,
            all intensities are a fraction of the incident beam.
    """
    # thresholding the image to get a rough estimate of the active area and
    # the non-active area.
    threshold_image = _detector_threshold(det_image)
    # find a value for the average background intensity.
    background = (1 - threshold_image) * det_image
    vacuum_intensity = background[background.nonzero()].mean()
    # create an image where all background is set to zero.
    active_layer = threshold_image * det_image

    # find the centre of the detector.
    # N.B. Currently this method will not work if the detector doesn't
    # fill at least half the image.

    m = centeredDistanceMatrix((det_image.shape[0] / 2, det_image.shape[1] / 2),
                               det_image)
    centre_image = np.multiply((m < 512), (1 - threshold_image))
    centre = scipy.ndimage.measurements.center_of_mass(centre_image)

    # Using the centre of the detector and sensitivity profile can be made.
    # The maximum gradient of this profile is the inner collection angle
    # of the detector.
    detector_sensitivity = _radial_profile(det_image, centre[::-1])
    grad = np.gradient(detector_sensitivity)
    ratio = inner_angle / np.argmax(grad)

    # Create a centre matrix around where the detector centre is found to be.
    d = centeredDistanceMatrix(centre, det_image)

    if outer_angle != 'None':
        # This limits the detector average value to only the region being,
        # illuminated by the beam.
        active_layer = np.multiply(active_layer, ((d * ratio) < outer_angle))
        threshold_image = np.multiply(threshold_image, ((d * ratio) < outer_angle))

    if flux_expo == 'None':
        # If no flux exponent is provided the detector sensitivity is simply
        # the average value of the active region.
        detector_intensity = active_layer[active_layer.nonzero()].mean()

    else:
        # Begin flux weighting detector method based on:
        # Martinez et al. Ultramicroscopy 2015, 159, 46–58.

        if flux_expo < 0:
            flux_expo = 0 - flux_expo

        # 1. Create a 2D flux profile from the exponent value given.
        flux = _func(d, 1, flux_expo, 0)
        # 2. Multiply 2D flux by the threshold image so that only angles within
        # active and illuminated area of the detector are considered.
        flux = np.multiply(flux, threshold_image)
        # 3. Normalise the 2D flux so that the mean intensity in the new flux
        # region is 1.
        flux = flux / flux[flux.nonzero()].mean()
        # 4. Multiply this 2D flux by the active layer of the detector.
        new_det = np.multiply(active_layer, flux)

        detector_intensity = new_det[new_det.nonzero()].mean()

    return (image - vacuum_intensity) / (detector_intensity - vacuum_intensity)
