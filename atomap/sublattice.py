import numpy as np
import scipy as sp
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import math
from scipy.spatial import cKDTree
import hyperspy.api as hs
from hyperspy.signals import Signal2D
import copy
from hyperspy.drawing._markers.point import Point

from atomap.tools import (
        _get_interpolated2d_from_unregular_data,
        project_position_property_sum_planes,
        array2signal2d, array2signal1d,
        Fingerprinter)

from atomap.plotting import (
        _make_atom_planes_marker_list, _make_atom_position_marker_list,
        _make_arrow_marker_list, _make_multidim_atom_plane_marker_list,
        _make_zone_vector_text_marker_list)
from atomap.atom_finding_refining import construct_zone_axes_from_sublattice

from atomap.atom_position import Atom_Position
from atomap.atom_plane import Atom_Plane

from atomap.external.add_marker import add_marker
from atomap.external.gaussian2d import Gaussian2D


class Sublattice():
    def __init__(
            self,
            atom_position_list,
            image,
            original_image=None,
            color='red',
            pixel_size=1.,
            ):
        """
        Parameters
        ----------
        atom_position_list : NumPy array
            In the form [[x0, y0], [x1, y1], [x2, y2], ... ]
        image : 2D NumPy array
        original_image : 2D NumPy array, optional
        color : string, optional
            Plotting color, default red.
        pixel_size : float, optional
            Scaling number, default 1.

        Attributes
        ----------
        x_position : list of floats
        y_position : list of floats
        sigma_x : list of floats
        sigma_y : list of floats
        sigma_average : list of floats
        rotation : list of floats
        ellipticity : list of floats

        Examples
        --------
        >>> import numpy as np
        >>> from atomap.sublattice import Sublattice
        >>> atom_positions = [[2, 2], [2, 4], [4, 2], [4, 4]]
        >>> image_data = np.random.random((7, 7))
        >>> sublattice = Sublattice(atom_positions, image_data)
        >>> s_sublattice = sublattice.get_atom_list_on_image()
        >>> s_sublattice.plot()

        More atom positions

        >>> x, y = np.mgrid[0:100:10j, 0:100:10j]
        >>> x, y = x.flatten(), y.flatten()
        >>> atom_positions = np.dstack((x, y))[0]
        >>> image_data = np.random.random((100, 100))
        >>> sublattice = Sublattice(atom_positions, image_data, color='yellow')
        >>> sublattice.get_atom_list_on_image(markersize=50).plot()
        """
        self.atom_list = []
        for atom_position in atom_position_list:
            atom = Atom_Position(atom_position[0], atom_position[1])
            self.atom_list.append(atom)
        self.zones_axis_average_distances = None
        self.zones_axis_average_distances_names = []
        self.atom_plane_list = []
        self.image = image
        if original_image is None:
            self.original_image = image
        else:
            self.original_image = original_image
        self.atom_planes_by_zone_vector = {}
        self._plot_clim = None
        self._tag = ''
        self.pixel_size = pixel_size
        self._plot_color = color
        self._pixel_separation = 0.0

    def __repr__(self):
        return '<%s, %s (atoms:%s,planes:%s)>' % (
            self.__class__.__name__,
            self._tag,
            len(self.atom_list),
            len(self.atom_planes_by_zone_vector),
            )

    @property
    def atom_positions(self):
        return([self.x_position, self.y_position])

    @property
    def x_position(self):
        x_pos = []
        for atom in self.atom_list:
            x_pos.append(atom.pixel_x)
        return(x_pos)

    @property
    def y_position(self):
        y_pos = []
        for atom in self.atom_list:
            y_pos.append(atom.pixel_y)
        return(y_pos)

    @property
    def sigma_x(self):
        sigma_x = []
        for atom in self.atom_list:
            sigma_x.append(abs(atom.sigma_x))
        return(sigma_x)

    @property
    def sigma_y(self):
        sigma_y = []
        for atom in self.atom_list:
            sigma_y.append(abs(atom.sigma_y))
        return(sigma_y)

    @property
    def sigma_average(self):
        sigma = np.array(self.sigma_x)+np.array(self.sigma_y)
        sigma *= 0.5
        return(sigma)

    @property
    def atom_amplitude_gaussian2d(self):
        amplitude = []
        for atom in self.atom_list:
            amplitude.append(atom.amplitude_gaussian)
        return(amplitude)

    @property
    def atom_amplitude_max_intensity(self):
        amplitude = []
        for atom in self.atom_list:
            amplitude.append(atom.amplitude_max_intensity)
        return(amplitude)

    @property
    def rotation(self):
        rotation = []
        for atom in self.atom_list:
            rotation.append(atom.rotation)
        return(rotation)

    @property
    def ellipticity(self):
        ellipticity = []
        for atom in self.atom_list:
            ellipticity.append(atom.ellipticity)
        return(ellipticity)

    @property
    def rotation_ellipticity(self):
        rotation_ellipticity = []
        for atom in self.atom_list:
            rotation_ellipticity.append(atom.rotation_ellipticity)
        return(rotation_ellipticity)

    def get_zone_vector_index(self, zone_vector_id):
        """Find zone vector index from zone vector name"""
        for zone_vector_index, zone_vector in enumerate(
                self.zones_axis_average_distances_names):
            if zone_vector == zone_vector_id:
                return(zone_vector_index)
        raise ValueError('Could not find zone_vector ' + str(zone_vector_id))

    def get_atom_angles_from_zone_vector(
            self,
            zone_vector0,
            zone_vector1,
            degrees=False):
        """
        Calculates for each atom in the sub lattice the angle
        between the atom, and the next atom in the atom planes
        in zone_vector0 and zone_vector1.
        Default will return the angles in radians.

        Parameters
        ----------
        zone_vector0 : tuple
            Vector for the first zone.
        zone_vector1 : tuple
            Vector for the second zone.
        degrees : bool, optional
            If True, will return the angles in degrees.
            Default False.
        """
        angle_list = []
        pos_x_list = []
        pos_y_list = []
        for atom in self.atom_list:
            angle = atom.get_angle_between_zone_vectors(
                    zone_vector0,
                    zone_vector1)
            if angle is not False:
                angle_list.append(angle)
                pos_x_list.append(atom.pixel_x)
                pos_y_list.append(atom.pixel_y)
        angle_list = np.array(angle_list)
        pos_x_list = np.array(pos_x_list)
        pos_y_list = np.array(pos_y_list)
        if degrees:
            angle_list = np.rad2deg(angle_list)

        return(pos_x_list, pos_y_list, angle_list)

    def get_atom_distance_list_from_zone_vector(
            self,
            zone_vector):
        """
        Get distance between each atom and the next atom in an
        atom plane given by the zone_vector. Returns the x- and
        y-position, and the distance between the atom and the
        monolayer. The position is set between the atom and the
        monolayer.

        For certain zone axes where there is several monolayers
        between the atoms in the atom plane, there will be some
        artifacts created. For example, in the perovskite (110)
        projection, the atoms in the <111> atom planes are
        separated by 3 monolayers.

        To avoid this problem use the function
        get_monolayer_distance_list_from_zone_vector.

        Parameters
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]
        atom_distance_list = []
        for atom_plane in atom_plane_list:
            dist = atom_plane.position_distance_to_neighbor()
            atom_distance_list.extend(dist)
        atom_distance_list = np.array(
                atom_distance_list).swapaxes(0, 1)
        return(
                atom_distance_list[0],
                atom_distance_list[1],
                atom_distance_list[2])

    def get_monolayer_distance_list_from_zone_vector(
            self,
            zone_vector):
        """
        Get distance between each atom and the next monolayer given
        by the zone_vector. Returns the x- and y-position, and the
        distance between the atom and the monolayer. The position
        is set between the atom and the monolayer.

        The reason for finding the distance between monolayer,
        instead of directly between the atoms is due to some zone axis
        having having a rather large distance between the atoms in
        one atom plane. For example, in the perovskite (110) projection,
        the atoms in the <111> atom planes are separated by 3 monolayers.
        This can give very bad results.

        To get the distance between atoms, use the function
        get_atom_distance_list_from_zone_vector.

        Parameters
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]
        x_list, y_list, z_list = [], [], []
        for index, atom_plane in enumerate(atom_plane_list[1:]):
            atom_plane_previous = atom_plane_list[index]
            plane_data_list = self.\
                _get_distance_and_position_list_between_atom_planes(
                    atom_plane_previous, atom_plane)
            x_list.extend(plane_data_list[0].tolist())
            y_list.extend(plane_data_list[1].tolist())
            z_list.extend(plane_data_list[2].tolist())
        return(x_list, y_list, z_list)

    def get_atom_distance_difference_from_zone_vector(
            self,
            zone_vector):
        """
        Get distance difference between atoms in atoms planes
        belonging to a zone axis.

        Parameters
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        x_list, y_list, z_list = [], [], []
        for atom_plane in self.atom_planes_by_zone_vector[zone_vector]:
            data = atom_plane.get_net_distance_change_between_atoms()
            if data is not None:
                x_list.extend(data[:, 0])
                y_list.extend(data[:, 1])
                z_list.extend(data[:, 2])
        return(x_list, y_list, z_list)

    def _property_position(
            self,
            property_list,
            x_position=None,
            y_position=None):
        if x_position is None:
            x_position = self.x_position
        if y_position is None:
            y_position = self.y_position
        data_list = np.array([
                        x_position,
                        y_position,
                        property_list])
        data_list = np.swapaxes(data_list, 0, 1)
        return(data_list)

    def _property_position_projection(
            self,
            interface_plane,
            property_list,
            x_position=None,
            y_position=None,
            scale_xy=1.0,
            scale_z=1.0):
        if x_position is None:
            x_position = self.x_position
        if y_position is None:
            y_position = self.y_position
        data_list = np.array([
                        x_position,
                        y_position,
                        property_list])
        data_list = np.swapaxes(data_list, 0, 1)
        line_profile_data = \
            project_position_property_sum_planes(
                data_list,
                interface_plane,
                rebin_data=True)
        line_profile_data = np.array(line_profile_data)
        position = line_profile_data[:, 0]*scale_xy
        data = line_profile_data[:, 1]*scale_z
        return(np.array([position, data]))

    def _get_regular_grid_from_unregular_property(
            self,
            x_list,
            y_list,
            z_list,
            upscale=2):
        """
        Interpolate unregularly spaced data points into a
        regularly spaced grid, useful for making data work
        with plotting using imshow.

        Parameters
        ----------
        x_list : list of numbers
            x-positions
        y_list : list of numbers
            y-positions
        z_list : list of numbers
            The property, for example distance between
            atoms, ellipticity or angle between atoms.
        """

        data_list = self._property_position(
            z_list,
            x_position=x_list,
            y_position=y_list)

        interpolate_x_lim = (0, self.image.shape[1])
        interpolate_y_lim = (0, self.image.shape[0])
        new_data = _get_interpolated2d_from_unregular_data(
            data_list,
            new_x_lim=interpolate_x_lim,
            new_y_lim=interpolate_y_lim,
            upscale=upscale)

        return(new_data)

    def _get_property_map(
            self,
            x_list,
            y_list,
            z_list,
            atom_plane_list=None,
            data_scale_z=1.0,
            add_zero_value_sublattice=None,
            upscale_map=2):
        data_scale = self.pixel_size
        if not(add_zero_value_sublattice is None):
            self._add_zero_position_to_data_list_from_atom_list(
                x_list,
                y_list,
                z_list,
                add_zero_value_sublattice.x_position,
                add_zero_value_sublattice.y_position)
        data_map = self._get_regular_grid_from_unregular_property(
            x_list,
            y_list,
            z_list)
        signal = array2signal2d(
                data_map[2], self.pixel_size/upscale_map, rotate_flip=True)
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                    atom_plane_list, scale=data_scale, add_numbers=False)
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def _get_property_line_profile(
            self,
            x_list,
            y_list,
            z_list,
            atom_plane,
            data_scale_xy=1.0,
            data_scale_z=1.0,
            invert_line_profile=False,
            add_markers=True,
            interpolate_value=50):
        """
        Project a 2 dimensional property to a plane, and get
        values as a function of distance from this plane.
        The function will attempt to combine the data points from
        the same monolayer, to get the property values as a function
        of each monolayer. The data will be returned as an interpolation
        of these values, since HyperSpy signals currently does not support
        non-linear axes.

        Parameters
        ----------
        x_list, y_list : Numpy 1D array
            x and y positions for z_list property value.
        z_list : Numpy 1D array
            The property value. x_list, y_list and z_list must have
            the same size.
        atom_plane : Atomap AtomPlane object
            The plane the data is projected onto.
        data_scale_xy : number, optional
            For scaling the x_list and y_list values.
        data_scale_z : number, optional
            For scaling the values in the z_list
        invert_line_profile : bool, optional, default False
            If True, will invert the x-axis values
        interpolate_value : int, default 50
            The amount of data points between in monolayer, due to
            HyperSpy signals not supporting non-linear axes.

        Returns
        -------
        HyperSpy signal1D

        Example
        -------
        >>> from numpy.random import random
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, random((9, 9)))
        >>> for atom in sublattice.atom_list:
        ...     atom.sigma_x, atom.sigma_y = 0.5*random()+1, 0.5*random()+1
        >>> sublattice.construct_zone_axes()
        >>> x = sublattice.x_position
        >>> y = sublattice.y_position
        >>> z = sublattice.ellipticity
        >>> plane = sublattice.atom_plane_list[20]
        >>> s = sublattice._get_property_line_profile(x, y, z, plane)
        >>> s.plot()
        """
        line_profile_data_list = self._property_position_projection(
            interface_plane=atom_plane,
            property_list=z_list,
            x_position=x_list,
            y_position=y_list,
            scale_xy=data_scale_xy,
            scale_z=data_scale_z)
        x_new = np.linspace(
                line_profile_data_list[0, 0],
                line_profile_data_list[0, -1],
                interpolate_value*len(line_profile_data_list[0]))
        y_new = np.interp(
                x_new,
                line_profile_data_list[0],
                line_profile_data_list[1],
                )
        if invert_line_profile:
            x_new *= -1
        data_scale = x_new[1]-x_new[0]
        offset = x_new[0]
        signal = array2signal1d(
                y_new,
                scale=data_scale,
                offset=offset)
        if add_markers:
            if invert_line_profile:
                x_list = line_profile_data_list[0]*-1
            else:
                x_list = line_profile_data_list[0]
            y_list = line_profile_data_list[1]
            marker_list = []
            for x, y in zip(x_list, y_list):
                marker_list.append(Point(x, y))
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def _get_pixel_separation(self, nearest_neighbors=2, leafsize=100):
        """
        Get the pixel separation by finding the distance between each atom
        and its two closest neighbors. From this distance list the median
        distance is found and divided by 2. This gives the distance used
        by for example atom_finding_refining.get_atom_positions

        Parameters
        ----------
        nearest_neighbor : int, optional, default 2

        Returns
        -------
        pixel_separation, int
        """
        atom_position_list = np.array(
                [self.x_position, self.y_position]).swapaxes(0, 1)
        nearest_neighbor_data = cKDTree(
                atom_position_list,
                leafsize=leafsize)
        distance_list = []
        for atom in self.atom_list:
            nn_data_list = nearest_neighbor_data.query(
                    atom.get_pixel_position(),
                    nearest_neighbors)
            # Skipping the first element,
            # since it points to the atom itself
            for nn_link in nn_data_list[1][1:]:
                distance = atom.get_pixel_distance_from_another_atom(
                        self.atom_list[nn_link])
                distance_list.append(distance)
        pixel_separation = np.median(distance_list)/2
        return(pixel_separation)

    def _find_nearest_neighbors(self, nearest_neighbors=9, leafsize=100):
        atom_position_list = np.array(
                [self.x_position, self.y_position]).swapaxes(0, 1)
        nearest_neighbor_data = sp.spatial.cKDTree(
                atom_position_list,
                leafsize=leafsize)
        for atom in self.atom_list:
            nn_data_list = nearest_neighbor_data.query(
                    atom.get_pixel_position(),
                    nearest_neighbors)
            nn_link_list = []
            # Skipping the first element,
            # since it points to the atom itself
            for nn_link in nn_data_list[1][1:]:
                nn_link_list.append(self.atom_list[nn_link])
            atom.nearest_neighbor_list = nn_link_list

    def get_atom_plane_slice_between_two_planes(
            self, atom_plane1, atom_plane2, zone_vector):
        atom_plane_start_index = None
        atom_plane_end_index = None
        for index, temp_atom_plane in enumerate(
                self.atom_planes_by_zone_vector[zone_vector]):
            if temp_atom_plane == atom_plane1:
                atom_plane_start_index = index
            if temp_atom_plane == atom_plane2:
                atom_plane_end_index = index
        if atom_plane_start_index > atom_plane_end_index:
            temp_index = atom_plane_start_index
            atom_plane_start_index = atom_plane_end_index
            atom_plane_end_index = temp_index
        atom_plane_slice = self.atom_planes_by_zone_vector[zone_vector][
                atom_plane_start_index:atom_plane_end_index]
        return(atom_plane_slice)

    def get_atom_list_between_four_atom_planes(
            self,
            par_atom_plane1,
            par_atom_plane2,
            ort_atom_plane1,
            ort_atom_plane2):
        ort_atom_plane_slice = self.get_atom_plane_slice_between_two_planes(
                ort_atom_plane1, ort_atom_plane2, ort_atom_plane1.zone_vector)
        par_atom_plane_slice = self.get_atom_plane_slice_between_two_planes(
                par_atom_plane1, par_atom_plane2, par_atom_plane1.zone_vector)

        par_atom_list = []
        for atom_plane in par_atom_plane_slice:
            par_atom_list.extend(atom_plane.atom_list)
        ort_atom_list = []
        for temp_atom_plane in ort_atom_plane_slice:
            temp_atom_list = []
            for atom in temp_atom_plane.atom_list:
                if atom in par_atom_list:
                    temp_atom_list.append(atom)
            ort_atom_list.extend(temp_atom_list)
        return(ort_atom_list)

    def _find_perpendicular_vector(self, v):
        if v[0] == 0 and v[1] == 0:
            raise ValueError('zero vector')
        return np.cross(v, [1, 0])

    def _sort_atom_planes_by_zone_vector(self):
        for zone_vector in self.zones_axis_average_distances:
            temp_atom_plane_list = []
            for atom_plane in self.atom_plane_list:
                if atom_plane.zone_vector == zone_vector:
                    temp_atom_plane_list.append(atom_plane)
            self.atom_planes_by_zone_vector[zone_vector] = temp_atom_plane_list

        for index, (zone_vector, atom_plane_list) in enumerate(
                self.atom_planes_by_zone_vector.items()):
            length = 100000000
            orthogonal_vector = (
                    length*zone_vector[1], -length*zone_vector[0])

            closest_atom_list = []
            for atom_plane in atom_plane_list:
                closest_atom = 10000000000000000000000000
                for atom in atom_plane.atom_list:
                    dist = atom.pixel_distance_from_point(
                        orthogonal_vector)
                    if dist < closest_atom:
                        closest_atom = dist
                closest_atom_list.append(closest_atom)
            atom_plane_list.sort(
                    key=dict(zip(atom_plane_list, closest_atom_list)).get)

    def _remove_bad_zone_vectors(self):
        zone_vector_delete_list = []
        for zone_vector in self.atom_planes_by_zone_vector:
            atom_planes = self.atom_planes_by_zone_vector[zone_vector]
            counter_atoms = 0
            for atom_plane in atom_planes:
                number_of_atoms = len(atom_plane.atom_list)
                if number_of_atoms == 2:
                    counter_atoms += 1
            ratio = counter_atoms/len(atom_planes)
            if ratio > 0.6:
                atom_planes_delete_list = []
                for atom_plane in atom_planes:
                    for atom in atom_plane.atom_list:
                        atom.in_atomic_plane.remove(atom_plane)
                    atom_planes_delete_list.append(atom_plane)
                for i, v in enumerate(self.zones_axis_average_distances):
                    if v == zone_vector:
                        self.zones_axis_average_distances_names.pop(i)
                zone_vector_delete_list.append(zone_vector)
                self.zones_axis_average_distances.remove(zone_vector)
                for atom_plane in atom_planes_delete_list:
                    self.atom_plane_list.remove(atom_plane)
        for zone_vector in zone_vector_delete_list:
            del self.atom_planes_by_zone_vector[zone_vector]

    def refine_atom_positions_using_2d_gaussian(
            self,
            image_data,
            percent_to_nn=0.40,
            rotation_enabled=True):
        for atom in tqdm(self.atom_list, desc="Gaussian fitting"):
            atom.refine_position_using_2d_gaussian(
                    image_data,
                    rotation_enabled=rotation_enabled,
                    percent_to_nn=percent_to_nn)

    def refine_atom_positions_using_center_of_mass(
            self, image_data, percent_to_nn=0.25):
        for atom in tqdm(self.atom_list, desc="Center of mass"):
            atom.refine_position_using_center_of_mass(
                image_data,
                percent_to_nn=percent_to_nn)

    def get_nearest_neighbor_directions(
            self, pixel_scale=True, neighbors=None):
        """
        Get the vector to the nearest neighbors for the atoms
        in the sublattice. Giving information similar to a FFT
        of the image, but for real space.

        Useful for seeing if the peakfinding and symmetry
        finder worked correctly. Potentially useful for
        doing structure fingerprinting.

        Parameters
        ----------
        pixel_scale : bool, optional. Default True
            If True, will return coordinates in pixel scale.
            If False, will return in data scale (pixel_size).
        neighbors : int, optional
            The number of neighbors returned for each atoms.
            If no number is given, will return all the neighbors,
            which is typcially 9 for each atom. As given when
            running the symmetry finder.

        Returns
        -------
        Position : tuple
            (x_position, y_position). Where both are numpy arrays.

        Examples
        --------
        >>> import numpy as np
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, np.random.random((9, 9)))
        >>> sublattice.construct_zone_axes()
        >>> x_pos, y_pos = sublattice.get_nearest_neighbor_directions()
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.scatter(x_pos, y_pos)

        With all the keywords

        >>> x_pos, y_pos = sublattice.get_nearest_neighbor_directions(
        ...     pixel_scale=False, neighbors=3)
        """
        if neighbors is None:
            neighbors = 10000

        x_pos_distances = []
        y_pos_distances = []
        for atom in self.atom_list:
            for index, neighbor_atom in enumerate(atom.nearest_neighbor_list):
                if index > neighbors:
                    break
                distance = atom.get_pixel_difference(neighbor_atom)
                if not ((distance[0] == 0) and (distance[1] == 0)):
                    x_pos_distances.append(distance[0])
                    y_pos_distances.append(distance[1])
        if not pixel_scale:
            scale = self.pixel_size
        else:
            scale = 1.
        x_pos_distances = np.array(x_pos_distances)*scale
        y_pos_distances = np.array(y_pos_distances)*scale
        return(x_pos_distances, y_pos_distances)

    def get_nearest_neighbor_directions_all(self):
        """
        Like get_nearest_neighbour_directions(), but considers
        all other atoms (instead of the typical 9) as neighbors
        from each atom.

        This method also does not require atoms to have the
        atom.nearest_neighbor_list parameter populated with
        sublattice._find_nearest_neighbors().

        Without the constraint of looking at only n nearest neighbours,
        blazing fast internal numpy functions can be utilized to
        calculate directions. However, memory usage will grow quadratically
        with the number of atomic columns. E.g.:
        1000 atomic columns will require ~8MB of memory.
        10,000 atomic columns will require ~800MB of memory.
        100,000 atomic columns will throw a MemoryError exception
        on most machines.

        Returns
        -------
        Pixel Position : np.array([x_position, y_position])

        Examples
        --------
        >>> import numpy as np
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, np.random.random((9, 9)))
        >>> x_pos, y_pos = sublattice.get_nearest_neighbor_directions_all()
        >>> mask = np.sqrt(x_pos**2 + y_pos**2) < 3
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.scatter(x_pos[mask], y_pos[mask])
        """

        n_atoms = len(self.atom_list)

        # Calculate the offset matrix
        #
        # Note: The terms 'direction', 'offset' and 'distance vector' are used
        # interchangeably in this method.
        x_array = np.asarray(self.x_position)
        y_array = np.asarray(self.y_position)
        dx = x_array - x_array[..., np.newaxis]
        dy = y_array - y_array[..., np.newaxis]
        offset = np.array([dx, dy])

        # Assert statements here are just to help the reader understand what's
        # going on by keeping track of the shapes of arrays used.
        assert offset.shape == (2, n_atoms, n_atoms)

        # Produce a mask that selects all elements except the diagonal
        # i.e. distance vectors from an atom to itself.
        mask = ~np.diag([True]*n_atoms)
        assert mask.shape == (n_atoms, n_atoms)

        # Remove the diagonal and flatten
        nn = np.array([offset[0][mask], offset[1][mask]])
        assert nn.shape == (2, n_atoms*(n_atoms-1))

        return nn

    def _make_translation_symmetry(self, pixel_separation_factor=7):
        pixel_radius = self._pixel_separation*pixel_separation_factor
        fp_2d = self.get_fingerprint_2d(pixel_radius=pixel_radius)
        clusters = []
        for zone_vector in fp_2d:
            cluster = (
                    float(format(zone_vector[0], '.2f')),
                    float(format(zone_vector[1], '.2f')))
            clusters.append(cluster)
        clusters = self._sort_vectors_by_length(clusters)
        clusters = self._remove_parallel_vectors(
                clusters,
                tolerance=self._pixel_separation/1.5)

        new_zone_vector_name_list = []
        for zone_vector in clusters:
            new_zone_vector_name_list.append(str(tuple(zone_vector)))

        self.zones_axis_average_distances = clusters
        self.zones_axis_average_distances_names = new_zone_vector_name_list

    def _sort_vectors_by_length(self, old_vector_list):
        vector_list = copy.deepcopy(old_vector_list)
        zone_vector_distance_list = []
        for vector in vector_list:
            distance = math.hypot(vector[0], vector[1])
            zone_vector_distance_list.append(distance)

        vector_list.sort(key=dict(zip(
            vector_list, zone_vector_distance_list)).get)
        return(vector_list)

    def _find_shortest_vector(self, vector_list):
        shortest_atom_distance = 100000000000000000000000000000
        for vector in vector_list:
            distance = math.hypot(vector[0], vector[1])
            if distance < shortest_atom_distance:
                shortest_atom_distance = distance
        return(shortest_atom_distance)

    def _remove_parallel_vectors(self, old_vector_list, tolerance=7):
        """
        Remove parallel and antiparallel zone vectors.
        """
        vector_list = copy.deepcopy(old_vector_list)
        element_prune_list = []
        for zone_index, zone_vector in enumerate(vector_list):
            for n in range(-4, 5):
                n_vector = (n*zone_vector[0], n*zone_vector[1])
                for temp_index, temp_zone_vector in enumerate(
                        vector_list[zone_index+1:]):
                    dist_x = temp_zone_vector[0]-n_vector[0]
                    dist_y = temp_zone_vector[1]-n_vector[1]
                    distance = math.hypot(dist_x, dist_y)
                    if distance < tolerance:
                        element_prune_list.append(zone_index+temp_index+1)
        element_prune_list = list(set(element_prune_list))
        element_prune_list.sort()
        element_prune_list.reverse()
        for element_prune in element_prune_list:
            del(vector_list[element_prune])
        return(vector_list)

    def _get_atom_plane_list_from_zone_vector(self, zone_vector):
        temp_atom_plane_list = []
        for atom_plane in self.atom_plane_list:
            if atom_plane.zone_vector == zone_vector:
                temp_atom_plane_list.append(atom_plane)
        return(temp_atom_plane_list)

    def _generate_all_atom_plane_list(self):
        for zone_vector in self.zones_axis_average_distances:
            self._find_all_atomic_planes_from_direction(zone_vector)

    def _find_all_atomic_planes_from_direction(self, zone_vector):
        for atom in self.atom_list:
            if not atom.is_in_atomic_plane(zone_vector):
                atom_plane = self._find_atomic_columns_from_atom(
                        atom, zone_vector)
                if not (len(atom_plane) == 1):
                    atom_plane_instance = Atom_Plane(
                            atom_plane, zone_vector, self)
                    for atom in atom_plane:
                        atom.in_atomic_plane.append(atom_plane_instance)
                    self.atom_plane_list.append(atom_plane_instance)

    def _find_atomic_columns_from_atom(
            self, start_atom, zone_vector, atom_range_factor=0.5):
        atom_range = atom_range_factor*self._pixel_separation
        end_of_atom_plane = False
        zone_axis_list1 = [start_atom]
        while not end_of_atom_plane:
            atom = zone_axis_list1[-1]
            atoms_within_distance = []
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = neighbor_atom.pixel_distance_from_point(
                        point=(
                            atom.pixel_x+zone_vector[0],
                            atom.pixel_y+zone_vector[1]))
                if distance < atom_range:
                    atoms_within_distance.append([distance, neighbor_atom])
            if atoms_within_distance:
                atoms_within_distance.sort()
                zone_axis_list1.append(atoms_within_distance[0][1])
            if zone_axis_list1[-1] is atom:
                end_of_atom_plane = True
                atom._end_atom.append(zone_vector)

        zone_vector2 = (-1*zone_vector[0], -1*zone_vector[1])
        start_of_atom_plane = False
        zone_axis_list2 = [start_atom]
        while not start_of_atom_plane:
            atom = zone_axis_list2[-1]
            atoms_within_distance = []
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = neighbor_atom.pixel_distance_from_point(
                        point=(
                            atom.pixel_x+zone_vector2[0],
                            atom.pixel_y+zone_vector2[1]))
                if distance < atom_range:
                    atoms_within_distance.append([distance, neighbor_atom])
            if atoms_within_distance:
                atoms_within_distance.sort()
                zone_axis_list2.append(atoms_within_distance[0][1])
            if zone_axis_list2[-1] is atom:
                start_of_atom_plane = True
                atom._start_atom.append(zone_vector)

        if not (len(zone_axis_list2) == 1):
            zone_axis_list1.extend(zone_axis_list2[1:])
        return(zone_axis_list1)

    def _find_missing_atoms_from_zone_vector(
            self, zone_vector, new_atom_tag=''):
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]

        new_atom_list = []
        new_atom_plane_list = []
        for atom_plane in atom_plane_list:
            temp_new_atom_list = []
            for atom_index, atom in enumerate(atom_plane.atom_list[1:]):
                previous_atom = atom_plane.atom_list[atom_index]
                difference_vector = previous_atom.get_pixel_difference(atom)
                new_atom_x = previous_atom.pixel_x -\
                    difference_vector[0]*0.5
                new_atom_y = previous_atom.pixel_y -\
                    difference_vector[1]*0.5
                new_atom = Atom_Position(new_atom_x, new_atom_y)
                new_atom._tag = new_atom_tag
                temp_new_atom_list.append(new_atom)
                new_atom_list.append((new_atom_x, new_atom_y))
            new_atom_plane_list.append(temp_new_atom_list)
        return(new_atom_list)

    def get_atom_planes_on_image(
            self, atom_plane_list, image=None, add_numbers=True, color='red'):
        """
        Get atom_planes signal as lines on the image.

        Parameters
        ----------
        atom_plane_list : list of atom_plane objects
            atom_planes to be plotted on the image contained in
        image : 2D Array, optional
        add_numbers : bool, optional, default True
            If True, will the number of the atom plane at the end of the
            atom plane line. Useful for finding the index of the atom plane.
        color : string, optional, default red
            The color of the lines and text used to show the atom planes.

        Returns
        -------
        HyperSpy signal2D object

        Examples
        --------
        >>> from numpy.random import random
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, random((9, 9)))
        >>> sublattice.construct_zone_axes()
        >>> zone_vector = sublattice.zones_axis_average_distances[0]
        >>> atom_planes = sublattice.atom_planes_by_zone_vector[zone_vector]
        >>> s = sublattice.get_atom_planes_on_image(atom_planes)
        >>> s.plot()
        """
        if image is None:
            image = self.original_image
        marker_list = _make_atom_planes_marker_list(
                atom_plane_list,
                add_numbers=add_numbers,
                scale=self.pixel_size,
                color=color)
        signal = array2signal2d(image, self.pixel_size)
        add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def get_all_atom_planes_by_zone_vector(
            self,
            zone_vector_list=None,
            image=None,
            add_numbers=True,
            color='red'):
        """
        Get a overview of atomic planes for some or all zone vectors.

        Parameters
        ----------
        zone_vector_list : optional
            List of zone vectors for visualizing atomic planes.
            Default is visualizing all the zone vectors.
        image : 2D Array, optional
        add_numbers : bool, optional, default True
            If True, will the number of the atom plane at the end of the
            atom plane line. Useful for finding the index of the atom plane.
        color : string, optional, default red
            The color of the lines and text used to show the atom planes.

        Returns
        -------
        HyperSpy signal2D object if given a single zone vector,
        list of HyperSpy signal2D if given a list (or none) of zone vectors.

        Examples
        --------

        Getting a list signals showing the atomic planes for all the
        zone vectors

        >>> from numpy.random import random
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, random((9, 9)))
        >>> sublattice.construct_zone_axes()
        >>> s = sublattice.get_all_atom_planes_by_zone_vector()
        >>> s.plot()

        Several zone vectors

        >>> zone_vec_list = sublattice.zones_axis_average_distances[0:3]
        >>> s = sublattice.get_all_atom_planes_by_zone_vector(zone_vec_list)
        >>> s.plot()

        Different image

        >>> im = random((9., 9))
        >>> s = sublattice.get_all_atom_planes_by_zone_vector(image=im)
        >>> s.plot()
        """
        if image is None:
            image = self.original_image
        if zone_vector_list is None:
            zone_vector_list = self.zones_axis_average_distances
        atom_plane_list = []
        for zone_vector in zone_vector_list:
            atom_plane_list.append(
                    self.atom_planes_by_zone_vector[zone_vector])
        marker_list = _make_multidim_atom_plane_marker_list(
                atom_plane_list, scale=self.pixel_size)
        signal = array2signal2d(image, self.pixel_size)
        signal = hs.stack([signal]*len(zone_vector_list))
        add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal.metadata.General.title = "Atom planes by zone vector"
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(image.shape[0]*0.1))
        y = signal_ax1.index2value(int(image.shape[1]*0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
                zone_vector_list, x=x, y=y)
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        return signal

    def get_atom_list_on_image(
            self,
            atom_list=None,
            image=None,
            color=None,
            add_numbers=False,
            markersize=20):
        """
        Plot atom positions on the image data.

        Parameters
        ----------
        atom_list : list of Atom objects, optional
            Atom positions to plot. If no list is given,
            will use the atom_list.
        image : 2-D numpy array, optional
            Image data for plotting. If none is given, will use
            the original_image.
        color : string, optional
        add_numbers : bool, default False
            Plot the number of the atom beside each atomic
            position in the plot. Useful for locating
            misfitted atoms.
        markersize : number, default 20
            Size of the atom position markers

        Returns
        -------
        HyperSpy 2D-signal
            The atom positions as permanent markers stored in the metadata.

        Examples
        --------
        >>> from numpy.random import random
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, random((9, 9)))
        >>> sublattice.construct_zone_axes()
        >>> s = sublattice.get_atom_list_on_image()
        >>> s.plot()

        Number all the atoms

        >>> s = sublattice.get_atom_list_on_image(add_numbers=True)
        >>> s.plot()

        Plot a subset of the atom positions

        >>> atoms = sublattice.atom_list[0:20]
        >>> s = sublattice.get_atom_list_on_image(
        ...     atom_list=atoms, add_numbers=True)
        >>> s.plot(cmap='viridis')

        Saving the signal as HyperSpy HDF5 file, which saves the atom
        positions as permanent markers.

        >>> s = sublattice.get_atom_list_on_image()
        >>> s.save("sublattice_atom_positions.hdf5", overwrite=True)
        """
        if color is None:
            color = self._plot_color
        if image is None:
            image = self.original_image
        if atom_list is None:
            atom_list = self.atom_list
        marker_list = _make_atom_position_marker_list(
                atom_list,
                scale=self.pixel_size,
                color=color,
                markersize=markersize,
                add_numbers=add_numbers)
        signal = array2signal2d(image, self.pixel_size)
        add_marker(signal, marker_list, permanent=True, plot_marker=False)

        return signal

    def _add_zero_position_to_data_list_from_atom_list(
            self,
            x_list,
            y_list,
            z_list,
            zero_position_x_list,
            zero_position_y_list):
        """
        Add zero value properties to position and property list.
        Useful to visualizing oxygen tilt pattern.

        Parameters
        ----------
        x_list : list of numbers
        y_list : list of numbers
        z_list : list of numbers
        zero_position_x_list : list of numbers
        zero_position_y_list : list of numbers
        """
        x_list.extend(zero_position_x_list)
        y_list.extend(zero_position_y_list)
        z_list.extend(np.zeros_like(zero_position_x_list))

    def get_ellipticity_vector(
            self,
            image=None,
            atom_plane_list=None,
            vector_scale=1.0,
            color='red'):
        """
        Visualize the ellipticity and direction of the atomic columns
        using markers in a HyperSpy signal.

        Parameters
        ----------
        image : 2-D Numpy array, optional
        atom_plane_list : list of AtomPlane instances
        vector_scale : scaling of the vector
        color : string

        Returns
        -------
        HyperSpy 2D-signal with the ellipticity vectors as
        permanent markers

        Examples
        --------
        >>> import numpy as np
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, np.random.random((9, 9)))
        >>> for atom in sublattice.atom_list:
        ...     atom.sigma_x, atom.sigma_y = 1., 1.2
        >>> s = sublattice.get_ellipticity_vector(vector_scale=20)
        >>> s.plot()
        """
        if image is None:
            image = self.original_image
        elli_list = []
        for atom in self.atom_list:
            elli_rot = atom.get_ellipticity_vector()
            elli_list.append([
                    atom.pixel_x,
                    atom.pixel_y,
                    elli_rot[0]*vector_scale,
                    elli_rot[1]*vector_scale,
                    ])
        signal = array2signal2d(image, self.pixel_size)
        marker_list = _make_arrow_marker_list(
                elli_list,
                scale=self.pixel_size,
                color=color)
        if atom_plane_list is not None:
            marker_list.extend(_make_atom_planes_marker_list(
                    atom_plane_list, scale=self.pixel_size, add_numbers=False))
        add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def get_atom_column_amplitude_max_intensity(
            self,
            image=None,
            percent_to_nn=0.40):
        if image is None:
            image = self.original_image

        percent_distance = percent_to_nn
        for atom in self.atom_list:
            atom.calculate_max_intensity(
                    image,
                    percent_to_nn=percent_distance)

    def get_atom_list_atom_amplitude_gauss2d_range(
            self,
            amplitude_range):
        atom_list = []
        for atom in self.atom_list:
            if atom.amplitude_gaussian > amplitude_range[0]:
                if atom.amplitude_gaussian < amplitude_range[1]:
                    atom_list.append(atom)
        return(atom_list)

    def save_map_from_datalist(
            self,
            data_list,
            data_scale,
            atom_plane=None,
            dtype='float32',
            signal_name="datalist_map.hdf5"):
        """data_list : numpy array, 4D"""
        im = hs.signals.Signal2D(data_list[2])
        x_scale = data_list[0][1][0] - data_list[0][0][0]
        y_scale = data_list[1][0][1] - data_list[1][0][0]
        im.axes_manager[0].scale = x_scale*data_scale
        im.axes_manager[1].scale = y_scale*data_scale
        im.change_dtype('float32')
        if not (atom_plane is None):
            im.metadata.add_node('marker.atom_plane.x')
            im.metadata.add_node('marker.atom_plane.y')
            im.metadata.marker.atom_plane.x =\
                atom_plane.get_x_position_list()
            im.metadata.marker.atom_plane.y =\
                atom_plane.get_y_position_list()
        im.save(signal_name, overwrite=True)

    def _get_distance_and_position_list_between_atom_planes(
            self,
            atom_plane0,
            atom_plane1):
        list_x, list_y, list_z = [], [], []
        for atom in atom_plane0.atom_list:
            pos_x, pos_y = atom_plane1.get_closest_position_to_point(
                    (atom.pixel_x, atom.pixel_y), extend_line=True)
            distance = atom.pixel_distance_from_point(
                    point=(pos_x, pos_y))
            list_x.append((pos_x + atom.pixel_x)*0.5)
            list_y.append((pos_y + atom.pixel_y)*0.5)
            list_z.append(distance)
        data_list = np.array([list_x, list_y, list_z])
        return(data_list)

    def get_ellipticity_line_profile(
            self,
            atom_plane,
            invert_line_profile=False,
            interpolate_value=50):
        signal = self._get_property_line_profile(
            self.x_position,
            self.y_position,
            self.ellipticity,
            atom_plane,
            data_scale_xy=self.pixel_size,
            invert_line_profile=invert_line_profile,
            interpolate_value=interpolate_value)
        return signal

    def get_ellipticity_map(
            self,
            upscale_map=2.0,
            atom_plane_list=None):
        """
        Get a HyperSpy signal showing the magnitude of the ellipticity
        for the sublattice.

        Parameters
        ----------
        upscale_map : number, default 2.0
            Amount of upscaling compared to the original image given
            to Atomap. Note, a high value here can greatly increase
            the memory use for large images.
        atom_plane_list : List of Atomap AtomPlane object, optional
            If a list of AtomPlanes are given, the plane positions
            will be added to the signal as permanent markers. Which
            can be visualized using s.plot(plot_markers=True).
            Useful for showing the location of for example an interface.

        Returns
        -------
        HyperSpy 2D signal

        Examples
        --------
        >>> from numpy.random import random
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, random((9, 9)))
        >>> for atom in sublattice.atom_list:
        ...     atom.sigma_x, atom.sigma_y = 0.5*random()+1, 0.5*random()+1
        >>> s_elli = sublattice.get_ellipticity_map()
        >>> s_elli.plot()

        Include an atom plane, which is added to the signal as a marker

        >>> sublattice.construct_zone_axes()
        >>> atom_plane = [sublattice.atom_plane_list[10]]
        >>> s_elli = sublattice.get_ellipticity_map(atom_plane_list=atom_plane)
        >>> s_elli.plot()
        """
        signal = self._get_property_map(
            self.x_position,
            self.y_position,
            self.ellipticity,
            atom_plane_list=atom_plane_list,
            upscale_map=upscale_map)
        title = 'Sublattice {} ellipticity'.format(self._tag)
        signal.metadata.General.title = title
        return signal

    def get_monolayer_distance_line_profile(
            self,
            zone_vector,
            atom_plane,
            invert_line_profile=False,
            interpolate_value=50):
        data_list = self.get_monolayer_distance_list_from_zone_vector(
                zone_vector)
        signal = self._get_property_line_profile(
                data_list[0],
                data_list[1],
                data_list[2],
                atom_plane,
                data_scale_xy=self.pixel_size,
                data_scale_z=self.pixel_size,
                invert_line_profile=invert_line_profile,
                interpolate_value=interpolate_value)
        return signal

    def get_monolayer_distance_map(
            self,
            zone_vector_list=None,
            atom_plane_list=None,
            upscale_map=2):
        zone_vector_index_list = self._get_zone_vector_index_list(
                zone_vector_list)
        zone_vector_list = []
        signal_list = []
        for zone_index, zone_vector in zone_vector_index_list:
            signal_title = 'Monolayer distance {}'.format(zone_index)
            data_list = self.get_monolayer_distance_list_from_zone_vector(
                    zone_vector)
            signal = self._get_property_map(
                data_list[0],
                data_list[1],
                data_list[2],
                upscale_map=upscale_map)
            signal.metadata.General.Title = signal_title
            signal_list.append(signal)
            zone_vector_list.append(zone_vector)

        if len(signal_list) == 1:
            signal = signal_list[0]
        else:
            signal = hs.stack(signal_list)
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                    atom_plane_list, scale=self.pixel_size, add_numbers=False)
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(signal_ax0.high_index*0.1))
        y = signal_ax1.index2value(int(signal_ax1.high_index*0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
                zone_vector_list, x=x, y=y)
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        title = 'Sublattice {} monolayer distance'.format(self._tag)
        signal.metadata.General.title = title
        return signal

    def get_atom_distance_map(
            self,
            zone_vector_list=None,
            atom_plane_list=None,
            data_scale_z=1.0,
            prune_outer_values=False,
            invert_line_profile=False,
            add_zero_value_sublattice=None,
            upscale_map=2):

        zone_vector_index_list = self._get_zone_vector_index_list(
                zone_vector_list)

        signal_list = []
        zone_vector_list = []
        for zone_index, zone_vector in zone_vector_index_list:
            signal_title = 'Atom distance {}'.format(zone_index)
            data_list = self.get_atom_distance_list_from_zone_vector(
                    zone_vector)

            signal = self._get_property_map(
                data_list[0],
                data_list[1],
                data_list[2],
                data_scale_z=data_scale_z,
                add_zero_value_sublattice=add_zero_value_sublattice,
                upscale_map=upscale_map)
            signal.metadata.General.Title = signal_title
            signal_list.append(signal)
            zone_vector_list.append(zone_vector)

        if len(signal_list) == 1:
            signal = signal_list[0]
        else:
            signal = hs.stack(signal_list)
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                    atom_plane_list, scale=self.pixel_size, add_numbers=False)
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(signal_ax0.high_index*0.1))
        y = signal_ax1.index2value(int(signal_ax1.high_index*0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
                zone_vector_list, x=x, y=y)
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        title = 'Sublattice {} atom distance'.format(self._tag)
        signal.metadata.General.title = title
        return signal

    def get_atom_distance_difference_line_profile(
            self,
            zone_vector,
            atom_plane,
            invert_line_profile=False,
            interpolate_value=50):
        data_list = self.get_atom_distance_difference_from_zone_vector(
                zone_vector)
        signal = self._get_property_line_profile(
                data_list[0],
                data_list[1],
                data_list[2],
                atom_plane,
                data_scale_xy=self.pixel_size,
                data_scale_z=self.pixel_size,
                invert_line_profile=invert_line_profile,
                interpolate_value=interpolate_value)
        return signal

    def get_atom_distance_difference_map(
            self,
            zone_vector_list=None,
            atom_plane_list=None,
            data_scale_z=1.0,
            prune_outer_values=False,
            invert_line_profile=False,
            add_zero_value_sublattice=None,
            upscale_map=2):
        zone_vector_index_list = self._get_zone_vector_index_list(
                zone_vector_list)
        zone_vector_list = []
        signal_list = []
        for zone_index, zone_vector in zone_vector_index_list:
            data_list = self.get_atom_distance_difference_from_zone_vector(
                    zone_vector)
            if len(data_list[2]) is not 0:
                signal = self._get_property_map(
                    data_list[0],
                    data_list[1],
                    data_list[2],
                    data_scale_z=data_scale_z,
                    add_zero_value_sublattice=add_zero_value_sublattice,
                    upscale_map=upscale_map)
                signal_list.append(signal)
                zone_vector_list.append(zone_vector)

        if len(signal_list) == 1:
            signal = signal_list[0]
        else:
            signal = hs.stack(signal_list)
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                    atom_plane_list, scale=self.pixel_size, add_numbers=False)
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(signal_ax0.high_index*0.1))
        y = signal_ax1.index2value(int(signal_ax1.high_index*0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
                zone_vector_list, x=x, y=y)
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        title = 'Sublattice {} atom distance difference'.format(self._tag)
        signal.metadata.General.title = title
        return signal

    def get_atom_model(self):
        model_image = np.zeros(self.image.shape)
        X, Y = np.meshgrid(np.arange(
            model_image.shape[1]), np.arange(model_image.shape[0]))

        g = Gaussian2D(
            centre_x=0.0,
            centre_y=0.0,
            sigma_x=1.0,
            sigma_y=1.0,
            rotation=1.0,
            A=1.0)

        for atom in tqdm(self.atom_list):
            g.A.value = atom.amplitude_gaussian
            g.centre_x.value = atom.pixel_x
            g.centre_y.value = atom.pixel_y
            g.sigma_x.value = atom.sigma_x
            g.sigma_y.value = atom.sigma_y
            g.rotation.value = atom.rotation
            model_image += g.function(X, Y)
        s = Signal2D(model_image)

        return(s)

    def _get_zone_vector_index_list(self, zone_vector_list):
        if zone_vector_list is None:
            zone_vector_list = self.zones_axis_average_distances

        zone_vector_index_list = []
        for zone_vector in zone_vector_list:
            for index, temp_zone_vector in enumerate(
                    self.zones_axis_average_distances):
                if temp_zone_vector == zone_vector:
                    zone_index = index
                    break
            zone_vector_index_list.append([zone_index, zone_vector])
        return(zone_vector_index_list)

    def _plot_debug_start_end_atoms(self):
        for zone_index, zone_vector in enumerate(
                self.zones_axis_average_distances):
            fig, ax = plt.subplots(figsize=(10, 10))
            cax = ax.imshow(self.image)
            if self._plot_clim:
                cax.set_clim(self._plot_clim[0], self._plot_clim[1])
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom._start_atom:
                    ax.plot(
                            atom.pixel_x,
                            atom.pixel_y,
                            'o', color='blue')
                    ax.text(
                            atom.pixel_x,
                            atom.pixel_y,
                            str(atom_index))
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom._end_atom:
                    ax.plot(
                            atom.pixel_x,
                            atom.pixel_y,
                            'o', color='green')
                    ax.text(
                            atom.pixel_x,
                            atom.pixel_y,
                            str(atom_index))
            ax.set_ylim(0, self.image.shape[0])
            ax.set_xlim(0, self.image.shape[1])
            fig.tight_layout()
            fig.savefig(
                    "debug_plot_start_end_atoms_zone" +
                    str(zone_index) + ".jpg")

    def _plot_atom_position_convergence(
            self, figname='atom_position_convergence.jpg'):
        position_absolute_convergence = []
        position_jump_convergence = []
        for atom in self.atom_list:
            dist0 = atom.get_position_convergence(
                    distance_to_first_position=True)
            dist1 = atom.get_position_convergence()
            position_absolute_convergence.append(dist0)
            position_jump_convergence.append(dist1)

        absolute_convergence = np.array(
                position_absolute_convergence).mean(axis=0)
        relative_convergence = np.array(
                position_jump_convergence).mean(axis=0)

        fig, axarr = plt.subplots(2, 1, sharex=True)
        absolute_ax = axarr[0]
        relative_ax = axarr[1]

        absolute_ax.plot(absolute_convergence)
        relative_ax.plot(relative_convergence)

        absolute_ax.set_ylabel("Average distance from start")
        relative_ax.set_ylabel("Average jump pr. iteration")
        relative_ax.set_xlabel("Refinement step")

        fig.tight_layout()
        fig.savefig(self._tag + "_" + figname)

    def get_zone_vector_mean_angle(self, zone_vector):
        """Get the mean angle between the atoms planes with a
        specific zone vector and the horizontal axis. For each
        atom plane the angle between all the atoms, its
        neighbor and horizontal axis is calculated.
        The mean of these angles for all the atom
        planes is returned.
        """
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]
        angle_list = []
        for atom_plane in atom_plane_list:
            temp_angle_list = atom_plane.get_angle_to_horizontal_axis()
            angle_list.extend(temp_angle_list)
        mean_angle = np.array(angle_list).mean()
        return(mean_angle)

    def construct_zone_axes(self, debug_plot=False, zone_axis_para_list=False):
        construct_zone_axes_from_sublattice(
                self,
                debug_plot=debug_plot,
                zone_axis_para_list=zone_axis_para_list)

    def _get_fingerprint(self, pixel_radius=100):
        """
        Produce a Fingerprint class object.

        Example
        -------
        >>> from numpy.random import random
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, random((9, 9)))
        >>> fp = sublattice._get_fingerprint()
        >>> fp_distance = fp.fingerprint_
        >>> fp_vector = fp.cluster_centers_
        """

        n_atoms = len(self.atom_list)

        # Get distance vectors to all neighbouring atoms from each atom
        x_pos, y_pos = self.get_nearest_neighbor_directions_all()

        # Assert statements here are just to help the reader understand the
        # algorithm by keeping track of the shapes of arrays used.
        assert x_pos.shape == y_pos.shape == (n_atoms*(n_atoms-1),)

        # Produce a mask that only select vectors that are shorter than radius
        mask = (x_pos**2 + y_pos**2) < pixel_radius**2
        assert mask.shape == (n_atoms*(n_atoms-1),)
        n_atoms_closer_than_radius = mask.sum()

        # Apply mask to get nearest neighbours
        nn = np.array([x_pos[mask], y_pos[mask]])
        assert nn.shape == (2, n_atoms_closer_than_radius,)

        # Apply the fingerprinter
        fingerprinter = Fingerprinter()
        fingerprinter.fit(nn.T)

        return fingerprinter

    def get_fingerprint_2d(self, pixel_radius=100):
        """
        Produce a distance and direction fingerprint of the sublattice.

        Example
        -------
        >>> import numpy as np
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, np.random.random((9, 9)))
        >>> fp = sublattice.get_fingerprint_2d()
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.scatter(fp[:,0], fp[:,1], marker='o')
        """
        fingerprinter = self._get_fingerprint(pixel_radius=pixel_radius)
        return fingerprinter.cluster_centers_

    def get_fingerprint_1d(self, pixel_radius=100):
        """
        Produce a distance fingerprint of the sublattice.

        Example
        -------
        >>> import numpy as np
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, np.random.random((9, 9)))
        >>> fp = sublattice.get_fingerprint_1d()
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.plot(fp, marker='o')
        """
        fingerprinter = self._get_fingerprint(pixel_radius=pixel_radius)
        return fingerprinter.fingerprint_

    def get_position_history(
            self,
            image=None,
            color='red',
            add_numbers=False,
            markersize=20):
        """
        Plot position history of each atom positions on the image data.

        Parameters
        ----------
        image : 2-D numpy array, optional
            Image data for plotting. If none is given, will use
            the original_image.
        color : string, default 'red'
        add_numbers : bool, default False
            Plot the number of the atom beside each atomic
            position in the plot. Useful for locating
            misfitted atoms.
        markersize : number, default 20
            Size of the atom position markers

        Returns
        -------
        HyperSpy 2D-signal
            The atom positions as permanent markers stored in the metadata.
        """
        if image is None:
            image = self.original_image

        pos_num = len(self.atom_list[0].old_pixel_x_list) + 1
        if pos_num == 1:
            s = self.get_atom_list_on_image(
                image=image,
                color=color,
                add_numbers=add_numbers,
                markersize=markersize)
            return(s)

        atom_num = len(self.atom_list)
        peak_list = np.zeros((pos_num, atom_num, 2))
        for i, atom in enumerate(self.atom_list):
            for j in range(len(atom.old_pixel_x_list)):
                peak_list[j, i, 0] = atom.old_pixel_x_list[j]
                peak_list[j, i, 1] = atom.old_pixel_y_list[j]
                peak_list[-1, i, 0] = atom.pixel_x
                peak_list[-1, i, 1] = atom.pixel_y

        signal = Signal2D(image)
        s = hs.stack([signal]*pos_num)

        marker_list_x = np.ones((len(peak_list), atom_num))*-100
        marker_list_y = np.ones((len(peak_list), atom_num))*-100

        for index, peaks in enumerate(peak_list):
            marker_list_x[index, 0:len(peaks)] = peaks[:, 0]
            marker_list_y[index, 0:len(peaks)] = peaks[:, 1]

        marker_list = []
        for i in trange(marker_list_x.shape[1]):
            m = hs.markers.point(
                    x=marker_list_x[:, i], y=marker_list_y[:, i], color='red')
            marker_list.append(m)

        s.add_marker(marker_list, permanent=True, plot_marker=False)
        return(s)
