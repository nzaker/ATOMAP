import numpy as np
import scipy
import matplotlib.pyplot as plt


def _detector_threshold(det_img):
    """Using an input detector image. The function thresholds returning a
    binary image.

    Parameters
    ----------
    det_img: np.array
            Detector map as 2D array.
    Returns
    -------
    threshold_img: np.array
            Boolean image demonstrating the active region of the detector.
    """

    threshold_img = np.zeros_like(det_img)
    threshold_img = (det_img - np.min(det_img)) / \
        (np.max(det_img) - np.min(det_img))
    threshold_img = (threshold_img >= 0.25)
    return threshold_img


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
            l = np.int(event.xdata)
            if l < self.right:
                self.left = l
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

    flux_pattern: np.array

    conv_angle: float

    Returns
    -------


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
    outer_cutoff = np.argmin(grad)

    # Plot the radial flux profile and allow the user to select the region for
    # power-law fitting.
    fig = plt.figure()
    plt.suptitle('Radial Profile of Electron Flux : Please select power-law region.',
                 fontsize=10)
    ax1 = fig.add_subplot(2, 1, 1)
    plt.plot(x, flux_profile)
    ax1.set_title('Radial Profile')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('Logarithmic Profile')
    profile = plt.plot(x, flux_profile)
    ax2.set_yscale('log')
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    profiler = InteractiveFluxAnalyser(profile, radius, flux_profile)

    return(profiler, flux_profile)


def find_exponent(coords, flux_profile, conv_angle):
    grad = np.gradient(flux_profile)
    x = np.array(range(flux_profile.shape[0]))
    radius = x * conv_angle / np.argmin(grad)
    xdata = radius[coords[0]:coords[1]]
    ydata = flux_profile[coords[0]:coords[1]]

    popt, pcov = scipy.optimize.curve_fit(_func, xdata, ydata, p0=([1, 1, 1]))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata, _func(xdata, *popt), 'r-', label='fit')
    ax1.set_yscale('log')
    plt.legend()
    plt.suptitle('Resulting fitted profile, fontsize=10')
    return(popt[1])


def detector_normalisation(img,
                           det_img,
                           inner_angle,
                           outer_angle,
                           flux_expo='None'):
    """Using an input detector image and flux image, a detector normalisation
    is applied with regards to

    Parameters
    ----------
    img: np.array
            Experimental image to be normalised.
    det_img: np.array
            Detector map as 2D array.
    flux_img: 'None' otherwise np.array
            Flux image is required for a flux weighted detector normalisation,
            if none is supplied the normal thresholing normalisation will be applied.

    Returns
    -------
    normalised_img: np.array
            The experimental image after detector normalisation sucha that,
            all intensities are a fraction of the incident beam.
    """
    # thresholding the image to get a rough estimate of the active area and
    # the non-active area.
    threshold_img = _detector_threshold(det_img)
    # find a value for the average background intensity.
    l = (1 - threshold_img) * det_img
    vacuum_intensity = l[l.nonzero()].mean()
    active_layer = threhold_img * det_img

    centre = scipy.ndimage.measurements.center_of_mass(threshold_img)
    detector_sensitivity = _radial_profile(det_img, centre[::-1])
    grad = np.gradient(detector_sensitivity)
    radius = np.array(range(detector_sensitivity.shape[0])) * \
        inner_angle / np.argmax(grad)

    if flux_expo == 'None':
        detector_intensity = active_layer[active_layer.nonzero()].mean()

    else:
        print('Oops! Sorry, I still need to add flux weighting method later!')

    return (img - vacuum_intensity)/(detector_intensity - vacuum_intensity)
