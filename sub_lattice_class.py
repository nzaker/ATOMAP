import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import hyperspy.api as hs
import copy
import json
from matplotlib.gridspec import GridSpec

from atomap_tools import \
        find_atom_position_1d_from_distance_list_and_atom_row,\
        _get_interpolated2d_from_unregular_data,\
        _get_clim_from_data,\
        find_atom_position_1d_from_distance_list_and_atom_row

from atomap_plotting import \
        plot_zone_vector_and_atom_distance_map,\
        plot_image_map_line_profile_using_interface_row,\
        _make_line_profile_subplot_from_three_parameter_data
from atomap_atom_finding_refining import get_peak2d_skimage

from atom_position_class import Atom_Position
from atom_row_class import Atom_Row

class Sub_Lattice():
    def __init__(self, atom_position_list, adf_image):
        self.atom_list = []
        for atom_position in atom_position_list:
            atom = Atom_Position(atom_position[0], atom_position[1])
            self.atom_list.append(atom)
        self.zones_axis_average_distances = None
        self.zones_axis_average_distances_names = []
        self.atom_row_list = []
        self.adf_image = adf_image
        self.original_adf_image = None
        self.atom_rows_by_zone_vector = {}
        self.plot_clim = None
        self.tag = ''
        self.save_path = "./"
        self.pixel_size = 1.0
        self.plot_color = 'blue'
        self.path_name = ""

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

    def get_atom_angles_from_zone_vector(
            self,
            zone_vector0,
            zone_vector1,
            degrees=False):
        """
        Calculates for each atom in the sub lattice the angle
        between the atom, and the next atom in the atom rows
        in zone_vector0 and zone_vector1.
        Default will return the angles in radians.

        Parameters:
        -----------
        zone_vector0 : tuple
        zone_vector1 : tuple
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
            if not (angle == False):
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
        atom row given by the zone_vector. Returns the x- and
        y-position, and the distance between the atom and the
        monolayer. The position is set between the atom and the
        monolayer.

        For certain zone axes where there is several monolayers
        between the atoms in the atom row, there will be some
        artifacts created. For example, in the perovskite (110)
        projection, the atoms in the <111> atom rows are
        separated by 3 monolayers.

        To avoid this problem use the function
        get_monolayer_distance_list_from_zone_vector.

        Parameter:
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        atom_row_list = self.atom_rows_by_zone_vector[zone_vector]
        atom_distance_list = []
        for atom_row in atom_row_list:
            atom_distance_list.extend(
                    atom_row.get_atom_distance_to_next_atom_and_position_list())
        atom_distance_list = np.array(atom_distance_list).swapaxes(0,1)
        return(atom_distance_list[0], atom_distance_list[1], atom_distance_list[2])

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
        one atom row. For example, in the perovskite (110) projection,
        the atoms in the <111> atom rows are separated by 3 monolayers.
        This can give very bad results.

        To get the distance between atoms, use the function
        get_atom_distance_list_from_zone_vector.

        Parameter:
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        atom_row_list = self.atom_rows_by_zone_vector[zone_vector]
        x_list, y_list, z_list = [], [], []
        for index, atom_row in enumerate(atom_row_list[1:]):
            atom_row_previous = atom_row_list[index]
            row_data_list = self._get_distance_and_position_list_between_atom_rows(
                    atom_row_previous, atom_row)
            x_list.extend(row_data_list[0].tolist())
            y_list.extend(row_data_list[1].tolist())
            z_list.extend(row_data_list[2].tolist())
        return(x_list, y_list, z_list)

    def get_atom_distance_difference_from_zone_vector(
            self,
            zone_vector):
        """
        Get distance difference between atoms in atoms rows
        belonging to a zone axis.

        Parameter:
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        x_list, y_list, z_list = [], [], []
        for atom_row in self.atom_rows_by_zone_vector[zone_vector]:
            data = atom_row.get_net_distance_change_between_atoms()
            if data is not None:
                x_list.extend(data[:,0])
                y_list.extend(data[:,1])
                z_list.extend(data[:,2])
        return(x_list, y_list, z_list)

    def get_property_and_positions(
            self,
            property_list,
            x_position=None,
            y_position=None):
        if x_position is None:
            x_position = self.x_position
        if y_position is None:
            y_position = self.y_position
        data_list = np.array(
                [x_position,
                y_position,
                property_list])
        data_list = np.swapaxes(data_list,0,1)
        return(data_list)

    def get_property_and_positions_atom_row_projection(
            self,
            interface_row,
            property_list,
            x_position=None,
            y_position=None,
            scale_xy=1.0,
            scale_z=1.0):
        if x_position is None:
            x_position = self.x_position
        if y_position is None:
            y_position = self.y_position
        data_list = np.array(
                [x_position,
                y_position,
                property_list])
        data_list = np.swapaxes(data_list,0,1)
        line_profile_data = \
                find_atom_position_1d_from_distance_list_and_atom_row(
            data_list,
            interface_row,
            rebin_data=True)
        line_profile_data = np.array(line_profile_data)
        position = line_profile_data[:,0]*scale_xy
        data = line_profile_data[:,1]*scale_z
        return(np.array([position, data]))

    def _get_regular_grid_from_unregular_property(
            self,
            x_list,
            y_list,
            z_list,
            upscale=4):
        """
        Interpolate unregularly spaced data points into a
        regularly spaced grid, useful for making data work
        with plotting using imshow.

        Parameters:
        -----------
        x_list : list of numbers
            x-positions
        y_list : list of numbers
            y-positions
        z_list : list of numbers
            The property, for example distance between
            atoms, ellipticity or angle between atoms.
        """

        data_list = self.get_property_and_positions(
            z_list,
            x_position=x_list,
            y_position=y_list)

        interpolate_x_lim = (0, self.adf_image.shape[1])
        interpolate_y_lim = (0, self.adf_image.shape[0])
        new_data = _get_interpolated2d_from_unregular_data(
            data_list,
            new_x_lim=interpolate_x_lim,
            new_y_lim=interpolate_y_lim,
            upscale=upscale)

        return(new_data)

    def plot_property_map_and_profile(
            self,
            x_list,
            y_list,
            z_list,
            interface_row=None,
            data_scale_z=1.0,
            save_signal=False,
            line_profile_prune_outer_values=False,
            invert_line_profile=False,
            add_zero_value_sublattice=None,
            figname="property_map_and_profile.jpg"):
        """
        Plot image data, property map and line profile. The property map
        can be any scalar 2-D array. The line profile is an integration
        of the property along a direction defined by an atom row.

        The property can be anything, for example distance between atoms
        along a zone axis, or ellipticity of the atoms.

        The image data is taken from the sub_lattice, either adf_image or
        original_adf_image.

        Parameters:
        -----------
        x_list : 1-D list of numbers
        y_list : 1-D list of numbers
        z_list : 1-D list of numbers
            The property list.
            x_list, y_list and z_list needs to have the same
            size.
        interface_row : Atom Row object, optional
            Atom row which is used when integrating the property,
            giving a line profile perpendicular to the interface_row.
            Will also be drawn on the image data and property map.
        data_scale_z : number, optional
            Scaling of the property list (z_list). Default 1.0.
        save_signal : bool, optional
            If true, will save the property map as a HyperSpy 2D signal.
            Default True
        line_profile_prune_outer_values : bool, optional
            If True, will prune the outer values of the line profile data.
            Useful for some datasets, where the low and high spatial
            positions in the line profile will be very noise due to
            low amount of averaging. Default False.
        invert_line_profile : bool, optional
            Reverts the spatial direction the line profile
            is plotted. Default False

        figname : string, optional
            Postfix in the image filename.
        """

        data_scale = self.pixel_size

        if interface_row is None:
            zone_vector = self.zones_axis_average_distances[0]
            middle_atom_row_index = int(len(self.atom_rows_by_zone_vector[zone_vector])/2)
            interface_row = self.atom_rows_by_zone_vector[zone_vector][middle_atom_row_index]

        line_profile_data_list = self.get_property_and_positions_atom_row_projection(
            interface_row=interface_row,
            property_list=z_list,
            x_position=x_list,
            y_position=y_list)
        line_profile_data_list = line_profile_data_list.swapaxes(0,1)

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

        if invert_line_profile is True:
            line_profile_data_list[:,0] *= -1

        clim = _get_clim_from_data(
                data_map[2]*data_scale_z,
                sigma=2,
                ignore_zeros=True,
                ignore_edges=True)

        if self.original_adf_image is None:
            image_data = self.adf_image
        else:
            image_data = self.original_adf_image

        plot_image_map_line_profile_using_interface_row(
            image_data,
            data_map,
            [line_profile_data_list],
            interface_row,
            data_scale=data_scale,
            data_scale_z=data_scale_z,
            clim=clim,
            rotate_atom_row_list_90_degrees=True,
            line_profile_prune_outer_values=line_profile_prune_outer_values,
            figname=self.save_path + self.tag + "_" + figname)

        if save_signal:
            save_signal_figname = figname[:-4]
            line_profile_dict = {
                    'position':(
                        line_profile_data_list[:,0]*data_scale).tolist(),
                    self.tag + '_position_difference':(
                        line_profile_data_list[:,1]*data_scale).tolist()}

            json_filename = self.save_path + self.tag +\
                    save_signal_figname + "_line_profile.json"
            with open(json_filename,'w') as fp:
                json.dump(line_profile_dict, fp)

            sig_name = self.save_path + self.tag +\
                    "_" + save_signal_figname + ".hdf5"
            self.save_map_from_datalist(
                data_map,
                data_scale,
                atom_row=interface_row,
                signal_name=sig_name)

    def find_nearest_neighbors(self, nearest_neighbors=9, leafsize=100):
        atom_position_list = np.array(
                [self.x_position, self.y_position]).swapaxes(0,1)
        nearest_neighbor_data = sp.spatial.cKDTree(
                atom_position_list,
                leafsize=leafsize)
        for atom in self.atom_list:
            nn_data_list = nearest_neighbor_data.query(
                    atom.get_pixel_position(),
                    nearest_neighbors)
            nn_link_list = []
            #Skipping the first element, since it points to the atom itself
            for nn_link in nn_data_list[1][1:]:
                nn_link_list.append(self.atom_list[nn_link])
            atom.nearest_neighbor_list = nn_link_list

    def get_atom_row_slice_between_two_rows(
            self, atom_row1, atom_row2, zone_vector):
        atom_row_start_index = None
        atom_row_end_index = None
        for index, temp_atom_row in enumerate(self.atom_rows_by_zone_vector[
            zone_vector]):
            if temp_atom_row == atom_row1:
                atom_row_start_index = index
            if temp_atom_row == atom_row2:
                atom_row_end_index = index
        if atom_row_start_index > atom_row_end_index:
            temp_index = atom_row_start_index
            atom_row_start_index = atom_row_end_index
            atom_row_end_index = temp_index
        atom_row_slice = self.atom_rows_by_zone_vector[zone_vector][
                atom_row_start_index:atom_row_end_index]
        return(atom_row_slice)

    def get_atom_list_between_four_atom_rows(
            self, par_atom_row1, par_atom_row2, ort_atom_row1, ort_atom_row2):
        ort_atom_row_slice = self.get_atom_row_slice_between_two_rows(
                ort_atom_row1, ort_atom_row2, ort_atom_row1.zone_vector)
        par_atom_row_slice = self.get_atom_row_slice_between_two_rows(
                par_atom_row1, par_atom_row2, par_atom_row1.zone_vector)

        par_atom_list = []
        for atom_row in par_atom_row_slice:
            par_atom_list.extend(atom_row.atom_list)
        ort_atom_list = []
        for temp_atom_row in ort_atom_row_slice:
            temp_atom_list = []
            for atom in temp_atom_row.atom_list:
                if atom in par_atom_list:
                    temp_atom_list.append(atom)
            ort_atom_list.extend(temp_atom_list)
        return(ort_atom_list)

    def _make_circular_mask(
            self, centerX, centerY, imageSizeX, imageSizeY, radius):
        y, x = np.ogrid[
                -centerX:imageSizeX-centerX, -centerY:imageSizeY-centerY]
        mask = x*x + y*y <= radius*radius
        return(mask)

    def find_perpendicular_vector(self, v):
        if v[0] == 0 and v[1] == 0:
            raise ValueError('zero vector')
        return np.cross(v, [1, 0])

    def _sort_atom_rows_by_zone_vector(self):
        for zone_vector in self.zones_axis_average_distances:
            temp_atom_row_list = []
            for atom_row in self.atom_row_list:
                if atom_row.zone_vector == zone_vector:
                    temp_atom_row_list.append(atom_row)
            self.atom_rows_by_zone_vector[zone_vector] = temp_atom_row_list

        for index, (zone_vector, atom_row_list) in enumerate(
                self.atom_rows_by_zone_vector.items()):
            length = 100000000
            orthogonal_vector = (
                    length*zone_vector[1], -length*zone_vector[0])

            closest_atom_list = []
            for atom_row in atom_row_list:
                closest_atom = 10000000000000000000000000
                for atom in atom_row.atom_list:
                    dist = atom.pixel_distance_from_point(
                        orthogonal_vector)
                    if dist < closest_atom:
                        closest_atom = dist
                closest_atom_list.append(closest_atom)
            atom_row_list.sort(
                    key=dict(zip(atom_row_list, closest_atom_list)).get)

    def refine_atom_positions_using_2d_gaussian(
            self,
            image_data,
            percent_distance_to_nearest_neighbor=0.40,
            rotation_enabled=True,
            debug_plot=False):
        for atom in self.atom_list:
            atom.refine_position_using_2d_gaussian(
                    image_data,
                    rotation_enabled=rotation_enabled,
                    percent_distance_to_nearest_neighbor=\
                        percent_distance_to_nearest_neighbor,
                    debug_plot=debug_plot)

    def refine_atom_positions_using_center_of_mass(
            self, image_data, percent_distance_to_nearest_neighbor=0.25):
        for atom_index, atom in enumerate(self.atom_list):
            atom.refine_position_using_center_of_mass(
                image_data,
                percent_distance_to_nearest_neighbor=\
                    percent_distance_to_nearest_neighbor)

    def get_nearest_neighbor_directions(self):
        x_pos_distances = []
        y_pos_distances = []
        for atom in self.atom_list:
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = atom.get_pixel_difference(neighbor_atom)
                if not ((distance[0] == 0) and (distance[1] == 0)):
                    x_pos_distances.append(distance[0])
                    y_pos_distances.append(distance[1])
        return(np.array([x_pos_distances, y_pos_distances]))

    def _make_nearest_neighbor_direction_distance_statistics(
            self,
            nearest_neighbor_histogram_max=0.8,
            debug_figname=''):
        x_pos_distances = []
        y_pos_distances = []
        for atom in self.atom_list:
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = atom.get_pixel_difference(neighbor_atom)
                if not ((distance[0] == 0) and (distance[1] == 0)):
                    x_pos_distances.append(distance[0])
                    y_pos_distances.append(distance[1])

        bins = (50, 50)
        histogram_range = nearest_neighbor_histogram_max/self.pixel_size
        direction_distance_intensity_hist = np.histogram2d(
                x_pos_distances,
                y_pos_distances,
                bins=bins,
                range=[
                    [-histogram_range, histogram_range],
                    [-histogram_range, histogram_range]])
        if not (debug_figname == ''):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(x_pos_distances, y_pos_distances)
            ax.set_ylim(-histogram_range, histogram_range)
            ax.set_xlim(-histogram_range, histogram_range)
            fig.savefig(self.save_path + debug_figname)

        hist_scale = direction_distance_intensity_hist[1][1] -\
            direction_distance_intensity_hist[1][0]

        s_direction_distance = hs.signals.Signal2D(
            direction_distance_intensity_hist[0])
        s_direction_distance.axes_manager[0].offset = -bins[0]/2
        s_direction_distance.axes_manager[1].offset = -bins[1]/2
        s_direction_distance.axes_manager[0].scale = hist_scale
        s_direction_distance.axes_manager[1].scale = hist_scale
        clusters = get_peak2d_skimage(
                s_direction_distance, separation=1)[0]

        shifted_clusters = []
        for cluster in clusters:
            temp_cluster = (
                    round((cluster[0]-bins[0]/2)*hist_scale, 2),
                    round((cluster[1]-bins[1]/2)*hist_scale, 2))
            shifted_clusters.append(temp_cluster)

        self.shortest_atom_distance = self._find_shortest_vector(
                shifted_clusters)
        shifted_clusters = self._sort_vectors_by_length(shifted_clusters)

        shifted_clusters = self._remove_parallel_vectors(
                shifted_clusters,
                tolerance=self.shortest_atom_distance/3.)

        hr_histogram = np.histogram2d(
                x_pos_distances,
                y_pos_distances,
                bins=(250, 250),
                range=[
                    [-histogram_range, histogram_range],
                    [-histogram_range, histogram_range]])

        new_zone_vector_list = self._refine_zone_vector_positions(
                shifted_clusters,
                hr_histogram,
                distance_percent=0.5)

        self.zones_axis_average_distances = new_zone_vector_list

        for new_zone_vector in new_zone_vector_list:
            self.zones_axis_average_distances_names.append(
                str(new_zone_vector))

    def _refine_zone_vector_positions(
            self,
            zone_vector_list,
            histogram,
            distance_percent=0.5):
        """ Refine zone vector positions using center of mass """
        scale = histogram[1][1] - histogram[1][0]
        offset = histogram[1][0]
        closest_distance = math.hypot(
                zone_vector_list[0][0],
                zone_vector_list[0][1])*distance_percent/scale

        new_zone_vector_list = []
        for zone_vector in zone_vector_list:
            zone_vector_x = (zone_vector[0]-offset)/scale
            zone_vector_y = (zone_vector[1]-offset)/scale
            circular_mask = self._make_circular_mask(
                    zone_vector_x,
                    zone_vector_y,
                    histogram[0].shape[0],
                    histogram[0].shape[1],
                    closest_distance)
            center_of_mass = ndimage.measurements.center_of_mass(
                    circular_mask*histogram[0])

            new_x_pos = float(
                    format(center_of_mass[0]*scale+offset, '.2f'))
            new_y_pos = float(
                    format(center_of_mass[1]*scale+offset, '.2f'))
            new_zone_vector_list.append((new_x_pos, new_y_pos))
        return(new_zone_vector_list)

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
        vector_list = copy.deepcopy(old_vector_list)
        element_prune_list = []
        for zone_index, zone_vector in enumerate(vector_list):
            opposite_vector = (-1*zone_vector[0], -1*zone_vector[1])
            for temp_index, temp_zone_vector in enumerate(
                    vector_list[zone_index+1:]):
                dist_x = temp_zone_vector[0]-opposite_vector[0]
                dist_y = temp_zone_vector[1]-opposite_vector[1]
                distance = math.hypot(dist_x, dist_y)
                if distance < tolerance:
                    element_prune_list.append(zone_index+temp_index+1)
        element_prune_list = list(set(element_prune_list))
        element_prune_list.sort()
        element_prune_list.reverse()
        for element_prune in element_prune_list:
            del(vector_list[element_prune])
        return(vector_list)

    def _get_atom_row_list_from_zone_vector(self, zone_vector):
        temp_atom_row_list = []
        for atom_row in self.atom_row_list:
            if atom_row.zone_vector == zone_vector:
                temp_atom_row_list.append(atom_row)
        return(temp_atom_row_list)

    def _generate_all_atom_row_list(self):
        for zone_vector in self.zones_axis_average_distances:
            self._find_all_atomic_rows_from_direction(zone_vector)

    def _find_all_atomic_rows_from_direction(self, zone_vector):
        for atom in self.atom_list:
            if not atom.is_in_atomic_row(zone_vector):
                atom_row = self._find_atomic_columns_from_atom(
                        atom, zone_vector)
                if not (len(atom_row) == 1):
                    atom_row_instance = Atom_Row(
                            atom_row, zone_vector, self)
                    for atom in atom_row:
                        atom.in_atomic_row.append(atom_row_instance)
                    self.atom_row_list.append(atom_row_instance)

    def _find_atomic_columns_from_atom(
            self, start_atom, zone_vector, atom_range_factor=0.5):
        atom_range = atom_range_factor*self.shortest_atom_distance
        end_of_atom_row = False
        zone_axis_list1 = [start_atom]
        while not end_of_atom_row:
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
                end_of_atom_row = True
                atom.end_atom.append(zone_vector)

        zone_vector2 = (-1*zone_vector[0], -1*zone_vector[1])
        start_of_atom_row = False
        zone_axis_list2 = [start_atom]
        while not start_of_atom_row:
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
                start_of_atom_row = True
                atom.start_atom.append(zone_vector)

        if not (len(zone_axis_list2) == 1):
            zone_axis_list1.extend(zone_axis_list2[1:])
        return(zone_axis_list1)

    def find_missing_atoms_from_zone_vector(
            self, zone_vector, new_atom_tag=''):
        atom_row_list = self.atom_rows_by_zone_vector[zone_vector]

        new_atom_list = []
        new_atom_row_list = []
        for atom_row in atom_row_list:
            temp_new_atom_list = []
            for atom_index, atom in enumerate(atom_row.atom_list[1:]):
                previous_atom = atom_row.atom_list[atom_index]
                difference_vector = previous_atom.get_pixel_difference(atom)
                new_atom_x = previous_atom.pixel_x -\
                    difference_vector[0]*0.5
                new_atom_y = previous_atom.pixel_y -\
                    difference_vector[1]*0.5
                new_atom = Atom_Position(new_atom_x, new_atom_y)
                new_atom.tag = new_atom_tag
                temp_new_atom_list.append(new_atom)
                new_atom_list.append((new_atom_x, new_atom_y))
            new_atom_row_list.append(temp_new_atom_list)
        return(new_atom_list)

    def plot_atom_row_on_stem_data(
            self, atom_row_list, figname="atom_row_plot.jpg"):
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(self.adf_image)
        if self.plot_clim:
            cax.set_clim(self.plot_clim[0], self.plot_clim[1])
        for atom_row in atom_row_list:
            x_pos = atom_row.get_x_position_list()
            y_pos = atom_row.get_y_position_list()
            ax.plot(x_pos, y_pos, 'o', color='blue')
        ax.set_ylim(0, self.adf_image.shape[0])
        ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(self.save_path + figname)

    def plot_atom_list_on_image_data(
            self,
            atom_list=None,
            image=None,
            plot_atom_numbers=False,
            markersize=2,
            fontsize=12,
            figsize=(10, 10),
            figdpi=200,
            figname="atom_plot.jpg",
            ax=None,
            fig=None):
        """
        Plot atom positions on the image data.

        Parameters:
        -----------
        atom_list : list of Atom objects, optional
            Atom positions to plot. If no list is given,
            will use the atom_list.
        image : 2-D numpy array, optional
            Image data for plotting. If none is given, will use
            the original_adf_image.
        plot_atom_numbers : bool, optional
            Plot the number of the atom beside each atomic
            position in the plot. Useful for locating
            misfitted atoms. Default False.
        markersize : number, optional
            Size of the atom position markers
        fontsize : int, optional
            If plot_atom_numbers is True, this will set
            the fontsize.
        figsize : tuple, optional
            Size of the figure in inches. Default (10, 10)
        figdpi : number, optional
            Pixels per inch for the figure. Default 200.
        figname : string, optional
            Name of the figure.

        Returns:
        --------
        Matplotlib figure object
        """
        if image is None:
            image = self.original_adf_image
        if atom_list is None:
            atom_list = self.atom_list
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(image, interpolation='nearest')
        if self.plot_clim:
            cax.set_clim(self.plot_clim[0], self.plot_clim[1])
        for atom_index, atom in enumerate(atom_list):
            ax.plot(
                    atom.pixel_x,
                    atom.pixel_y,
                    'o', markersize=markersize,
                    color='blue')
            if plot_atom_numbers:
                ax.text(
                        atom.pixel_x,
                        atom.pixel_y,
                        str(atom_index),
                        fontsize=fontsize,
                        color='red')
        ax.set_ylim(0, image.shape[0])
        ax.set_xlim(0, image.shape[1])
        fig.tight_layout()
        fig.savefig(self.save_path + figname, dpi=figdpi)
        return(fig)

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

        Parameters:
        -----------
        x_list : list of numbers
        y_list : list of numbers
        z_list : list of numbers
        zero_position_x_list : list of numbers
        zero_position_y_list : list of numbers
        """
        x_list.extend(zero_position_x_list)
        y_list.extend(zero_position_y_list)
        z_list.extend(np.zeros_like(zero_position_x_list))

    def plot_rotation(self, interface_row=None, clim=None, figname=''):
        x_pos_list, y_pos_list, x_rot_list, y_rot_list = [], [], [], []
        for atom in self.atom_list:
            x_pos_list.append(atom.pixel_x)
            y_pos_list.append(atom.pixel_y)
            rot = atom.get_rotation_vector()
            x_rot_list.append(rot[0])
            y_rot_list.append(rot[1])

        image_data = self.original_adf_image

        fig, axarr = plt.subplots(2, 1, figsize=(10, 20))

        image_ax = axarr[0]
        rot_ax = axarr[1]

        image_ax.imshow(image_data)
        image_ax.set_ylim(0, image_data.shape[0])
        image_ax.set_xlim(0, image_data.shape[1])

        rot_ax.quiver(
                x_pos_list,
                y_pos_list,
                x_rot_list,
                y_rot_list,
                scale=40.0,
                headwidth=0.0,
                headlength=0.0,
                headaxislength=0.0,
                pivot='middle')
        rot_ax.imshow(image_data, alpha=0.0)
        rot_ax.set_xlim(min(x_pos_list), max(x_pos_list))
        rot_ax.set_ylim(min(y_pos_list), max(y_pos_list))
        figname = self.save_path + self.tag + "_rotation"
        fig.tight_layout()
        fig.savefig(figname + ".png", dpi=200)

    def plot_ellipticity_rotation(
            self, interface_row=None, clim=None, figname=''):
        x_pos_list, y_pos_list, x_rot_list, y_rot_list = [], [], [], []
        for atom in self.atom_list:
            x_pos_list.append(atom.pixel_x)
            y_pos_list.append(atom.pixel_y)
            rot = atom.get_ellipticity_vector()
            x_rot_list.append(rot[0])
            y_rot_list.append(rot[1])

        image_data = self.original_adf_image

        fig, axarr = plt.subplots(2, 1, figsize=(10, 20))

        image_ax = axarr[0]
        rot_ax = axarr[1]

        image_ax.imshow(image_data)
        image_ax.set_ylim(0, image_data.shape[0])
        image_ax.set_xlim(0, image_data.shape[1])

        rot_ax.quiver(
                x_pos_list,
                y_pos_list,
                x_rot_list,
                y_rot_list,
                scale=40.0,
                headwidth=0.0,
                headlength=0.0,
                headaxislength=0.0,
                pivot='middle')
        rot_ax.imshow(image_data, alpha=0.0)
        rot_ax.set_xlim(min(x_pos_list), max(x_pos_list))
        rot_ax.set_ylim(min(y_pos_list), max(y_pos_list))
        figname = self.save_path + self.tag + "_ellipticity_rotation"
        fig.tight_layout()
        fig.savefig(figname + ".png", dpi=300)

    def plot_all_atom_rows(self, fignameprefix="atom_row"):
        for zone_index, zone_vector in enumerate(
                self.zones_axis_average_distances):
            fig, ax = plt.subplots(figsize=(20, 20))
            cax = ax.imshow(self.adf_image)
            if self.plot_clim:
                cax.set_clim(self.plot_clim[0], self.plot_clim[1])
            for atom_row_index, atom_row in enumerate(
                    self.atom_rows_by_zone_vector[zone_vector]):
                x_pos = atom_row.get_x_position_list()
                y_pos = atom_row.get_y_position_list()
                ax.plot(x_pos, y_pos, lw=3, color='blue')
                ax.text(
                        atom_row.start_atom.pixel_x,
                        atom_row.start_atom.pixel_y,
                        str(atom_row_index),
                        color='red')
            ax.set_ylim(0, self.adf_image.shape[0])
            ax.set_xlim(0, self.adf_image.shape[1])
            ax.set_title(str(zone_index) + " , " + str(zone_vector))
            fig.tight_layout()
            fig.savefig(
                    self.save_path +
                    fignameprefix +
                    str(zone_index) + ".jpg")

    def get_atom_column_amplitude_gaussian2d(
            self,
            image=None,
            percent_distance_to_nearest_neighbor=0.40):
        if image is None:
            image = self.original_adf_image

        percent_distance = percent_distance_to_nearest_neighbor
        for atom in self.atom_list:
            atom.fit_2d_gaussian_with_mask_centre_locked(
                    image,
                    percent_distance_to_nearest_neighbor=percent_distance)

    def get_atom_column_amplitude_max_intensity(
            self,
            image=None,
            percent_distance_to_nearest_neighbor=0.40):
        if image is None:
            image = self.original_adf_image

        percent_distance = percent_distance_to_nearest_neighbor
        for atom in self.atom_list:
            atom.calculate_max_intensity(
                    image,
                    percent_distance_to_nearest_neighbor=percent_distance)

    def get_atom_list_atom_amplitude_gauss2d_range(
            self,
            amplitude_range):
        atom_list = []
        for atom in self.atom_list:
            if atom.amplitude_gaussian > amplitude_range[0]:
                if atom.amplitude_gaussian < amplitude_range[1]:
                    atom_list.append(atom)
        return(atom_list)

    def get_atom_list_atom_sigma_range(
            self,
            sigma_range):
        atom_list = []
        for atom in self.atom_list:
            if atom.sigma_average > sigma_range[0]:
                if atom.sigma_average < sigma_range[1]:
                    atom_list.append(atom)
        return(atom_list)

    def plot_atom_column_hist_sigma_maps(
            self,
            bins=10,
            markersize=1,
            figname="atom_sigma_hist_gauss2d_map.jpg"):
        counts, bin_sizes = np.histogram(self.sigma_average, bins=bins)
        fig, axarr = plt.subplots(
                1, len(bin_sizes),
                figsize=(5*len(bin_sizes), 5))
        ax_hist = axarr[0]
        ax_hist.hist(self.sigma_average, bins=bins)
        for index, ax in enumerate(axarr[1:]):
            ax.imshow(self.original_adf_image)
            atom_list = self.get_atom_list_atom_sigma_range(
                    (bin_sizes[index], bin_sizes[index+1]))
            for atom in atom_list:
                ax.plot(
                        atom.pixel_x, atom.pixel_y,
                        'o', markersize=markersize)
            ax.set_ylim(0, self.adf_image.shape[0])
            ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(figname)

    def plot_atom_column_hist_amplitude_gauss2d_maps(
            self,
            bins=10,
            markersize=1,
            figname="atom_amplitude_gauss2d_hist_map.jpg"):
        counts, bin_sizes = np.histogram(
                self.atom_amplitude_gaussian2d, bins=bins)
        fig, axarr = plt.subplots(
                1, len(bin_sizes),
                figsize=(5*len(bin_sizes), 5))
        ax_hist = axarr[0]
        ax_hist.hist(self.atom_amplitude_gaussian2d, bins=bins)
        for index, ax in enumerate(axarr[1:]):
            ax.imshow(self.original_adf_image)
            atom_list = self.get_atom_list_atom_amplitude_gauss2d_range(
                    (bin_sizes[index], bin_sizes[index+1]))
            for atom in atom_list:
                ax.plot(atom.pixel_x, atom.pixel_y, 'o', markersize=markersize)
            ax.set_ylim(0, self.adf_image.shape[0])
            ax.set_xlim(0, self.adf_image.shape[1])
        fig.tight_layout()
        fig.savefig(figname)

    def plot_atom_column_histogram_sigma(
            self,
            bins=20,
            figname="atom_amplitude_sigma_histogram.png"):
        fig, ax = plt.subplots()
        ax.hist(
                self.sigma_average,
                bins=bins)
        ax.set_xlabel("Intensity bins")
        ax.set_ylabel("Amount")
        ax.set_title("Atom sigma average histogram, Gaussian2D")
        fig.savefig(figname, dpi=200)

    def plot_atom_column_histogram_amplitude_gauss2d(
            self,
            bins=20,
            xlim=None,
            figname="atom_amplitude_gauss2d_histogram.png"):
        fig, ax = plt.subplots()
        ax.hist(
                self.atom_amplitude_gaussian2d,
                bins=bins)
        if not (xlim is None):
            ax.set_xlim(xlim[0], xlim[1])
        ax.set_xlabel("Intensity bins")
        ax.set_ylabel("Amount")
        ax.set_title("Atom amplitude histogram, Gaussian2D")
        fig.savefig(figname, dpi=200)

    def plot_atom_column_histogram_max_intensity(
            self,
            bins=20,
            figname="atom_amplitude_max_intensity_histogram.png"):
        fig, ax = plt.subplots()
        ax.hist(
                self.atom_amplitude_max_intensity,
                bins=bins)
        ax.set_xlabel("Intensity bins")
        ax.set_ylabel("Amount")
        ax.set_title("Atom amplitude histogram, max intensity")
        fig.savefig(figname, dpi=200)

    def plot_amplitude_sigma_scatter(
            self,
            figname="sigma_amplitude_scatter.png"):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self.sigma_average, self.atom_amplitude_gaussian2d)
        ax.set_xlabel("Average sigma")
        ax.set_ylabel("Amplitude")
        ax.set_title("Sigma and amplitude scatter")
        fig.savefig(figname, dpi=200)

    def plot_amplitude_sigma_hist2d(
            self,
            bins=30,
            figname="sigma_amplitude_hist2d.png"):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist2d(
                self.sigma_average,
                self.atom_amplitude_gaussian2d,
                bins=bins)
        ax.set_xlabel("Average sigma")
        ax.set_ylabel("Amplitude")
        ax.set_title("Sigma and amplitude hist2d")
        fig.savefig(figname, dpi=200)

    def save_map_from_datalist(
            self,
            data_list,
            data_scale,
            atom_row=None,
            dtype='float32',
            signal_name="datalist_map.hdf5"):
        """data_list : numpy array, 4D"""
        im = hs.signals.Signal2D(data_list[2])
        x_scale = data_list[0][1][0] - data_list[0][0][0]
        y_scale = data_list[1][0][1] - data_list[1][0][0]
        im.axes_manager[0].scale = x_scale*data_scale
        im.axes_manager[1].scale = y_scale*data_scale
        im.change_dtype('float32')
        if not (atom_row is None):
            im.metadata.add_node('marker.atom_row.x')
            im.metadata.add_node('marker.atom_row.y')
            im.metadata.marker.atom_row.x =\
                atom_row.get_x_position_list()
            im.metadata.marker.atom_row.y =\
                atom_row.get_y_position_list()
        im.save(signal_name, overwrite=True)

    def _get_distance_and_position_list_between_atom_rows(
            self,
            atom_row0,
            atom_row1):
        list_x, list_y, list_z = [], [], []
        for atom in atom_row0.atom_list:
            pos_x, pos_y = atom_row1.get_closest_position_to_point(
                    (atom.pixel_x, atom.pixel_y), extend_line=True)
            distance = atom.pixel_distance_from_point(
                    point=(pos_x, pos_y))
            list_x.append((pos_x + atom.pixel_x)*0.5)
            list_y.append((pos_y + atom.pixel_y)*0.5)
            list_z.append(distance)
        data_list = np.array([list_x, list_y, list_z])
        return(data_list)

    def plot_ellipticity_map(
            self,
            interface_row=None,
            save_signal=False,
            data_scale_z=1.0,
            line_profile_prune_outer_values=False,
            invert_line_profile=False,
            figname="ellipticity.jpg"):
        """
        Plot a map and line profile of the ellipticity.
        """
        self.plot_property_map_and_profile(
            self.x_position,
            self.y_position,
            self.ellipticity,
            interface_row=interface_row,
            data_scale_z=data_scale_z,
            save_signal=save_signal,
            line_profile_prune_outer_values=
                line_profile_prune_outer_values,
            invert_line_profile=invert_line_profile,
            figname=figname)

    def plot_atom_distance_difference_map(
            self,
            zone_vector_list=None,
            interface_row=None,
            save_signal=False,
            data_scale_z=1.0,
            line_profile_prune_outer_values=False,
            invert_line_profile=False,
            add_zero_value_sublattice=None,
            figname="atom_distance_difference.jpg"):
        """
        Plot a map and line profile of the difference between atoms.
        """
        zone_vector_index_list = self._get_zone_vector_index_list(
                zone_vector_list)

        for zone_index, zone_vector in zone_vector_index_list:
            tempname = "zone" + str(zone_index) + "_" + figname

            data_list = self.get_atom_distance_difference_from_zone_vector(
                    zone_vector)
            self.plot_property_map_and_profile(
                data_list[0],
                data_list[1],
                data_list[2],
                interface_row=interface_row,
                data_scale_z=data_scale_z,
                save_signal=save_signal,
                line_profile_prune_outer_values=
                    line_profile_prune_outer_values,
                invert_line_profile=invert_line_profile,
                add_zero_value_sublattice=add_zero_value_sublattice,
                figname=tempname)

    def plot_monolayer_distance_map(
            self,
            zone_vector_list=None,
            interface_row=None,
            save_signal=False,
            data_scale_z=1.0,
            line_profile_prune_outer_values=False,
            invert_line_profile=False,
            figname="monolayer_distance.jpg"):

        zone_vector_index_list = self._get_zone_vector_index_list(
                zone_vector_list)

        for zone_index, zone_vector in zone_vector_index_list:
            tempname = "zone" + str(zone_index) + "_" + figname

            data_list = self.get_monolayer_distance_list_from_zone_vector(
                    zone_vector)
            self.plot_property_map_and_profile(
                data_list[0],
                data_list[1],
                data_list[2],
                interface_row=interface_row,
                data_scale_z=data_scale_z,
                save_signal=save_signal,
                line_profile_prune_outer_values=
                    line_profile_prune_outer_values,
                invert_line_profile=invert_line_profile,
                figname=tempname)

    def plot_atom_distance_map(
            self,
            zone_vector_list=None,
            interface_row=None,
            save_signal=False,
            data_scale_z=1.0,
            line_profile_prune_outer_values=False,
            invert_line_profile=False,
            figname="atom_distance.jpg"):

        zone_vector_index_list = self._get_zone_vector_index_list(
                zone_vector_list)

        for zone_index, zone_vector in zone_vector_index_list:
            tempname = "zone" + str(zone_index) + "_" + figname

            data_list = self.get_atom_distance_list_from_zone_vector(
                    zone_vector)
            self.plot_property_map_and_profile(
                data_list[0],
                data_list[1],
                data_list[2],
                interface_row=interface_row,
                data_scale_z=data_scale_z,
                save_signal=save_signal,
                line_profile_prune_outer_values=
                    line_profile_prune_outer_values,
                invert_line_profile=invert_line_profile,
                figname=tempname)

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
            cax = ax.imshow(self.adf_image)
            if self.plot_clim:
                cax.set_clim(self.plot_clim[0], self.plot_clim[1])
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom.start_atom:
                    ax.plot(
                            atom.pixel_x,
                            atom.pixel_y,
                            'o', color='blue')
                    ax.text(
                            atom.pixel_x,
                            atom.pixel_y,
                            str(atom_index))
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom.end_atom:
                    ax.plot(
                            atom.pixel_x,
                            atom.pixel_y,
                            'o', color='green')
                    ax.text(
                            atom.pixel_x,
                            atom.pixel_y,
                            str(atom_index))
            ax.set_ylim(0, self.adf_image.shape[0])
            ax.set_xlim(0, self.adf_image.shape[1])
            fig.tight_layout()
            fig.savefig(
                    self.save_path +
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
        fig.savefig(self.save_path + self.tag + "_" + figname)

    def plot_parameter_line_profiles(
            self,
            interface_row,
            parameter_list=[],
            parameter_name_list=None,
            zone_vector_number_list=None,
            x_lim=False,
            extra_line_marker_list=[],
            invert_line_profiles=False,
            figname=None):
        if zone_vector_number_list is None:
            zone_vector_number_list = range(7)

        number_of_subplots = len(zone_vector_number_list) +\
            len(parameter_list)

        figsize = (15, number_of_subplots*3)
        fig = plt.figure(figsize=figsize)

        gs = GridSpec(10*number_of_subplots, 10)

        line_profile_gs_size = 10
        line_profile_index = 0
        for zone_vector_number in zone_vector_number_list:
            zone_vector = self.zones_axis_average_distances[zone_vector_number]
            data_list = self.get_distance_difference_data_list_for_zone_vector(
                zone_vector)
            data_list = np.swapaxes(np.array(data_list), 0, 1)

            ax = fig.add_subplot(
                    gs[
                        line_profile_index*line_profile_gs_size:
                        (line_profile_index+1)*line_profile_gs_size, :])

            _make_line_profile_subplot_from_three_parameter_data(
                    ax,
                    data_list,
                    interface_row,
                    scale_x=self.pixel_size,
                    scale_y=self.pixel_size,
                    invert_line_profiles=invert_line_profiles)

            zone_vector_name = self.zones_axis_average_distances_names[
                    zone_vector_number]
            ylabel = self.tag + ", " + zone_vector_name + \
                "\nPosition deviation, [nm]"
            ax.set_ylabel(ylabel)

            line_profile_index += 1

        for index, parameter in enumerate(parameter_list):
            data_list = self.get_property_and_positions(parameter)

            ax = fig.add_subplot(
                    gs[
                        line_profile_index*line_profile_gs_size:
                        (line_profile_index+1)*line_profile_gs_size, :])

            _make_line_profile_subplot_from_three_parameter_data(
                    ax,
                    data_list,
                    interface_row,
                    scale_x=self.pixel_size,
                    invert_line_profiles=invert_line_profiles)

            if not (parameter_name_list is None):
                ax.set_ylabel(parameter_name_list[index])

            line_profile_index += 1

        if x_lim is False:
            x_min = 100000000000
            x_max = -10000000000
            for ax in fig.axes:
                ax_xlim = ax.get_xlim()
                if ax_xlim[0] < x_min:
                    x_min = ax_xlim[0]
                if ax_xlim[1] > x_max:
                    x_max = ax_xlim[1]
            for ax in fig.axes:
                ax.set_xlim(x_min, x_max)
        else:
            for ax in fig.axes:
                ax.set_xlim(x_lim[0], x_lim[1])

        for extra_line_marker in extra_line_marker_list:
            for ax in fig.axes:
                ax.axvline(extra_line_marker, color='red')

        fig.tight_layout()
        if figname is None:
            figname = self.save_path + self.tag + "_lattice_line_profiles.jpg"
        fig.savefig(figname, dpi=100)
