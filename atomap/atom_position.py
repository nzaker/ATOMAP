import copy
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
from atomap.tools import _make_circular_mask
from scipy import ndimage
import math
from atomap.external.gaussian2d import Gaussian2D


class Atom_Position:
    def __init__(self, x, y):
        self.pixel_x = x
        self.pixel_y = y
        self.nearest_neighbor_list = None
        self.in_atomic_plane = []
        self.start_atom = []
        self.end_atom = []
        self.atom_planes = []
        self.tag = ''
        self.old_pixel_x_list = []
        self.old_pixel_y_list = []
        self.sigma_x = 1.0
        self.sigma_y = 1.0
        self.rotation = 0.01
        self.amplitude_gaussian = 1.0
        self.amplitude_max_intensity = 1.0

    def __repr__(self):
        return '<%s, %s (x:%s,y:%s,sx:%s,sy:%s,r:%s,e:%s)>' % (
            self.__class__.__name__,
            self.tag,
            round(self.pixel_x, 1), round(self.pixel_y, 1),
            round(self.sigma_x, 1), round(self.sigma_y, 1),
            round(self.rotation, 1), round(self.ellipticity, 1),
            )

    @property
    def sigma_x(self):
        return(self.__sigma_x)

    @sigma_x.setter
    def sigma_x(self, new_sigma_x):
        self.__sigma_x = abs(new_sigma_x)

    @property
    def sigma_y(self):
        return(self.__sigma_y)

    @sigma_y.setter
    def sigma_y(self, new_sigma_y):
        self.__sigma_y = abs(new_sigma_y)

    @property
    def sigma_average(self):
        sigma = (abs(self.sigma_x)+abs(self.sigma_y))*0.5
        return(sigma)

    @property
    def rotation(self):
        return(self.__rotation)

    @rotation.setter
    def rotation(self, new_rotation):
        self.__rotation = new_rotation % math.pi

    @property
    def rotation_ellipticity(self):
        """Rotation between the "x-axis" and longest sigma.
        Basically giving the direction of the ellipticity."""
        if self.sigma_x > self.sigma_y:
            temp_rotation = self.__rotation % math.pi
        else:
            temp_rotation = (self.__rotation+(math.pi/2)) % math.pi
        return(temp_rotation)

    @property
    def ellipticity(self):
        """Largest sigma divided by the shortest"""
        if self.sigma_x > self.sigma_y:
            return(self.sigma_x/self.sigma_y)
        else:
            return(self.sigma_y/self.sigma_x)

    def get_pixel_position(self):
        return((self.pixel_x, self.pixel_y))

    def get_pixel_difference(self, atom):
        """Vector between self and given atom"""
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        return((x_distance, y_distance))

    def get_angle_between_atoms(self, atom0, atom1=None):
        """
        Returns the angle between itself and two atoms
        in radians, or between another atom and the
        horizontal axis.

        Parameters:
        -----------
        atom0 : Atom Position object
        atom1 : Atom Position object, optional
            If atom1 is not specified, the angle between
            itself, atom0 and the horizontal axis will be
            returned.

        Returns:
        Angle in radians
        """
        vector0 = np.array([
            atom0.pixel_x - self.pixel_x,
            atom0.pixel_y - self.pixel_y])
        if atom1 is None:
            vector1 = np.array([
                self.pixel_x+1000,
                0])
        else:
            vector1 = np.array([
                atom1.pixel_x - self.pixel_x,
                atom1.pixel_y - self.pixel_y])
        cosang = np.dot(vector0, vector1)
        sinang = np.linalg.norm(np.cross(vector0, vector1))
        return(np.arctan2(sinang, cosang))

    def get_angle_between_zone_vectors(
            self,
            zone_vector0,
            zone_vector1):
        """
        Return the angle between itself and the next atoms in
        the atom planes belonging to zone_vector0 and zone_vector1
        """
        atom0 = self.get_next_atom_in_zone_vector(zone_vector0)
        atom1 = self.get_next_atom_in_zone_vector(zone_vector1)
        if atom0 is False:
            return(False)
        if atom1 is False:
            return(False)
        angle = self.get_angle_between_atoms(atom0, atom1)
        return(angle)

    def get_image_slice_around_atom(
            self,
            image_data,
            slice_size):
        """
        Return a square slice of the image data, with the
        atom position in the center.

        Parameters
        ----------
        image_data : Numpy 2D array
        slice_size : int
            Width and height of the square slice

        Returns
        -------
        2D numpy array
        """
        x0 = self.pixel_x - slice_size/2
        x1 = self.pixel_x + slice_size/2
        y0 = self.pixel_y - slice_size/2
        y1 = self.pixel_y + slice_size/2

        if x0 < 0.0:
            x0 = 0
        if y0 < 0.0:
            y0 = 0
        if x1 > image_data.shape[1]:
            x1 = image_data.shape[1]
        if y1 > image_data.shape[0]:
            x1 = image_data.shape[0]

        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        
        data_slice = copy.deepcopy(image_data[y0:y1, x0:x1])
        return(data_slice)

    def _plot_gaussian2d_debug(
            self,
            slice_radius,
            gaussian,
            data_slice):

            X, Y = np.meshgrid(
                    np.arange(-slice_radius, slice_radius, 1),
                    np.arange(-slice_radius, slice_radius, 1))
            s_m = gaussian.function(X, Y)

            fig, axarr = plt.subplots(2, 2)
            ax0 = axarr[0][0]
            ax1 = axarr[0][1]
            ax2 = axarr[1][0]
            ax3 = axarr[1][1]

            ax0.imshow(data_slice, interpolation="nearest")
            ax1.imshow(s_m, interpolation="nearest")
            ax2.plot(data_slice.sum(0))
            ax2.plot(s_m.sum(0))
            ax3.plot(data_slice.sum(1))
            ax3.plot(s_m.sum(1))

            fig.tight_layout()
            fig.savefig(
                "debug_plot_2d_gaussian_" +
                str(np.random.randint(1000, 10000)) + ".jpg", dpi=400)
            plt.close('all')

    def get_closest_neighbor(self):
        """
        Find the closest neighbor to an atom in the same sub lattice.

        Returns
        -------
        Atomap atom_position object
        """
        closest_neighbor = 100000000000000000
        for neighbor_atom in self.nearest_neighbor_list:
            distance = self.get_pixel_distance_from_another_atom(
                    neighbor_atom)
            if distance < closest_neighbor:
                closest_neighbor = distance
        return(closest_neighbor)

    def calculate_max_intensity(
            self,
            image_data,
            percent_to_nn=0.40):
        """ If the Gaussian is centered outside the masked area,
        this function returns False"""
        plt.ioff()
        closest_neighbor = self.get_closest_neighbor()

        slice_size = closest_neighbor * percent_to_nn * 2
        data_slice = self.get_image_slice_around_atom(
                image_data, slice_size)

        data_slice_max = data_slice.max()
        self.amplitude_max_intensity = data_slice_max

        return(data_slice_max)

    def fit_2d_gaussian_with_mask_centre_locked(
            self,
            image_data,
            rotation_enabled=True,
            percent_to_nn=0.40,
            debug_plot=False):
        """ If the Gaussian is centered outside the masked area,
        this function returns False"""
        plt.ioff()
        closest_neighbor = self.get_closest_neighbor()

        slice_size = closest_neighbor * percent_to_nn * 2
        data_slice = self.get_image_slice_around_atom(
                image_data, slice_size)
        slice_radius = slice_size/2

        data_slice_max = data_slice.max()
        data = data_slice

        mask = _make_circular_mask(
                slice_radius,
                slice_radius,
                data.shape[0],
                data.shape[1],
                closest_neighbor*percent_to_nn)
        data = copy.deepcopy(data)
        mask = np.invert(mask)
        data[mask] = 0
        g = Gaussian2D(
                centre_x=0.0,
                centre_y=0.0,
                sigma_x=self.sigma_x,
                sigma_y=self.sigma_y,
                rotation=self.rotation,
                A=data_slice_max)

        s = hs.signals.Signal2D(data)
        s.axes_manager[0].offset = -slice_radius
        s.axes_manager[1].offset = -slice_radius
        s = hs.stack([s]*2)
        m = s.create_model()
        m.append(g)
        g.centre_x.free = False
        g.centre_y.free = False
        if rotation_enabled:
            g.rotation.free = True
        else:
            g.rotation.free = False
        m.fit()

        if debug_plot:
            self._plot_gaussian2d_debug(
                    slice_radius,
                    g,
                    data)

        self.amplitude_gaussian = g.A.value
        return(g)

    def fit_2d_gaussian_with_mask(
            self,
            image_data,
            rotation_enabled=True,
            percent_to_nn=0.40,
            debug_plot=False):
        """ If the Gaussian is centered outside the masked area,
        this function returns False"""
        plt.ioff()
        closest_neighbor = self.get_closest_neighbor()

        slice_size = closest_neighbor * percent_to_nn * 2
        data_slice = self.get_image_slice_around_atom(
                image_data, slice_size)
        slice_radius = slice_size/2

        data_slice -= data_slice.min()
        data_slice_max = data_slice.max()
        data = data_slice

        mask = _make_circular_mask(
                slice_radius,
                slice_radius,
                data.shape[0],
                data.shape[1],
                closest_neighbor*percent_to_nn)
        data = copy.deepcopy(data)
        mask = np.invert(mask)
        data[mask] = 0
        g = Gaussian2D(
                centre_x=0.0,
                centre_y=0.0,
                sigma_x=self.sigma_x,
                sigma_y=self.sigma_y,
                rotation=self.rotation,
                A=data_slice_max)

        if rotation_enabled:
            g.rotation.free = True
        else:
            g.rotation.free = False

        s = hs.signals.Signal2D(data)
        s.axes_manager[0].offset = -slice_radius
        s.axes_manager[1].offset = -slice_radius
        s = hs.stack([s]*2)
        m = s.create_model()
        m.append(g)
        m.fit()

        if debug_plot:
            self._plot_gaussian2d_debug(
                    slice_radius,
                    g,
                    data)

        # If the Gaussian centre is located outside the masked region,
        # return False
        dislocation = math.hypot(g.centre_x.value, g.centre_y.value)
        if dislocation > slice_radius:
            return(False)
        else:
            g.centre_x.value += self.pixel_x
            g.centre_y.value += self.pixel_y
            return(g)

    def refine_position_using_2d_gaussian(
            self,
            image_data,
            rotation_enabled=True,
            percent_to_nn=0.40,
            debug_plot=False):
        """
        Parameters
        ----------
        image_data : Numpy 2D array
        rotation_enabled : bool, optional
            If True, the Gaussian will be able to rotate.
            Note, this can increase the chance of fitting failure.
            Default True.
        percent_to_nn : float, optional
            The percent of the distance to the nearest neighbor atom
            in the same sub lattice. The distance times this percentage
            defines the mask around the atom where the Gaussian will be
            fitted. A smaller value can reduce the effect from
            neighboring atoms, but might also decrease the accuracy of
            the fitting due to less data to fit to.
            Default 0.4 (40%).
        debug_plot : bool, optional
            Make debug figure for every Gaussian fit.
            Useful for debugging failed Gaussian fitting.
            Default False.
        """

        for i in range(10):
            g = self.fit_2d_gaussian_with_mask(
                image_data,
                rotation_enabled=rotation_enabled,
                percent_to_nn=percent_to_nn,
                debug_plot=debug_plot)
            if g is False:
                print("Fitting missed")
                if i == 9:
                    new_x, new_y = self.get_center_position_com(
                        image_data,
                        percent_to_nn=percent_to_nn)
                    new_sigma_x = self.sigma_x
                    new_sigma_y = self.sigma_y
                    new_rotation = self.rotation
                    break
                else:
                    percent_to_nn *= 0.95
            else:
                new_x = g.centre_x.value
                new_y = g.centre_y.value
                new_rotation = g.rotation.value % math.pi
                new_sigma_x = abs(g.sigma_x.value)
                new_sigma_y = abs(g.sigma_y.value)
                break

        self.old_pixel_x_list.append(self.pixel_x)
        self.old_pixel_y_list.append(self.pixel_y)

        self.pixel_x = new_x
        self.pixel_y = new_y

        self.rotation = new_rotation
        self.sigma_x = new_sigma_x
        self.sigma_y = new_sigma_y
        if g is not False:
            self.amplitude_gaussian = g.A.value
        else:
            self.amplitude_gaussian = 0.0

    def get_center_position_com(
            self,
            image_data,
            percent_to_nn=0.40):
        closest_neighbor = 100000000000000000
        for neighbor_atom in self.nearest_neighbor_list:
            distance = self.get_pixel_distance_from_another_atom(
                    neighbor_atom)
            if distance < closest_neighbor:
                closest_neighbor = distance
        mask = _make_circular_mask(
                self.pixel_y,
                self.pixel_x,
                image_data.shape[0],
                image_data.shape[1],
                closest_neighbor*percent_to_nn)
        data = copy.deepcopy(image_data)
        mask = np.invert(mask)
        data[mask] = 0

        center_of_mass = self._calculate_center_of_mass(data)

        new_x, new_y = center_of_mass[1], center_of_mass[0]
        return(new_x, new_y)

    def refine_position_using_center_of_mass(
            self,
            image_data,
            percent_to_nn=0.40):
        new_x, new_y = self.get_center_position_com(
                image_data,
                percent_to_nn)
        self.old_pixel_x_list.append(self.pixel_x)
        self.old_pixel_y_list.append(self.pixel_y)
        self.pixel_x = new_x
        self.pixel_y = new_y

    def _calculate_center_of_mass(self, data):
        center_of_mass = ndimage.measurements.center_of_mass(data)
        return(center_of_mass)

    def get_atomic_plane_from_zone_vector(self, zone_vector):
        for atomic_plane in self.in_atomic_plane:
            if atomic_plane.zone_vector[0] == zone_vector[0]:
                if atomic_plane.zone_vector[1] == zone_vector[1]:
                    return(atomic_plane)
        return(False)

    def get_neighbor_atoms_in_atomic_plane_from_zone_vector(
            self, zone_vector):
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        atom_plane_atom_neighbor_list = []
        for atom in self.nearest_neighbor_list:
            if atom in atom_plane.atom_list:
                atom_plane_atom_neighbor_list.append(atom)
        return(atom_plane_atom_neighbor_list)

    def is_in_atomic_plane(self, zone_direction):
        for atomic_plane in self.in_atomic_plane:
            if atomic_plane.zone_vector[0] == zone_direction[0]:
                if atomic_plane.zone_vector[1] == zone_direction[1]:
                    return(True)
        return(False)

    def get_ellipticity_vector(self):
        elli = self.ellipticity - 1
        rot = self.get_rotation_vector()
        vector = (elli*rot[0], elli*rot[1])
        return(vector)

    def get_rotation_vector(self):
        rot = self.rotation
        vector = (
                math.cos(rot),
                math.sin(rot))
        return(vector)

    def get_pixel_distance_from_another_atom(self, atom):
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        total_distance = math.hypot(x_distance, y_distance)
        return(total_distance)

    def pixel_distance_from_point(self, point=(0, 0)):
        dist = math.hypot(
                self.pixel_x - point[0], self.pixel_y - point[1])
        return(dist)

    def get_index_in_atom_plane(self, atom_plane):
        for atom_index, atom in enumerate(atom_plane.atom_list):
            if atom == self:
                return(atom_index)

    def get_next_atom_in_atom_plane(self, atom_plane):
        current_index = self.get_index_in_atom_plane(atom_plane)
        if self == atom_plane.end_atom:
            return(False)
        else:
            next_atom = atom_plane.atom_list[current_index+1]
            return(next_atom)

    def get_previous_atom_in_atom_plane(self, atom_plane):
        current_index = self.get_index_in_atom_plane(atom_plane)
        if self == atom_plane.start_atom:
            return(False)
        else:
            previous_atom = atom_plane.atom_list[current_index-1]
            return(previous_atom)

    def get_next_atom_in_zone_vector(self, zone_vector):
        """Get the next atom in the atom plane belonging to
        zone vector"""
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        if atom_plane is False:
            return(False)
        next_atom = self.get_next_atom_in_atom_plane(atom_plane)
        return(next_atom)

    def get_previous_atom_in_zone_vector(self, zone_vector):
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        if atom_plane is False:
            return(False)
        previous_atom = self.get_previous_atom_in_atom_plane(atom_plane)
        return(previous_atom)

    def can_atom_plane_be_reached_through_zone_vector(
            self, atom_plane, zone_vector):
        for test_atom_plane in self.atom_planes:
            if test_atom_plane.zone_vector == zone_vector:
                for temp_atom in test_atom_plane.atom_list:
                    for temp_atom_plane in temp_atom.atom_planes:
                        if temp_atom_plane == atom_plane:
                            return(test_atom_plane)
        return(False)

    def get_position_convergence(
            self, distance_to_first_position=False):
        x_list = self.old_pixel_x_list
        y_list = self.old_pixel_y_list
        distance_list = []
        for index, (x, y) in enumerate(zip(x_list[1:], y_list[1:])):
            if distance_to_first_position:
                previous_x = x_list[0]
                previous_y = y_list[0]
            else:
                previous_x = x_list[index]
                previous_y = y_list[index]
            dist = math.hypot(x - previous_x, y - previous_y)
            distance_list.append(dist)
        return(distance_list)
