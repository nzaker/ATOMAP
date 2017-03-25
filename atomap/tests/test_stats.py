import matplotlib
matplotlib.use('Agg')
import unittest
import numpy as np
from atomap.sublattice import Sublattice
from atomap.stats import (
        plot_amplitude_sigma_hist2d,
        plot_atom_column_hist_amplitude_gauss2d_maps,
        plot_atom_column_histogram_sigma,
        plot_atom_column_histogram_amplitude_gauss2d,
        plot_atom_column_histogram_max_intensity,
        plot_amplitude_sigma_scatter,
        plot_amplitude_sigma_hist2d,
        get_atom_list_atom_sigma_range,
        )


class test_stats(unittest.TestCase):

    def setUp(self):
        self.atoms_N = 10
        image_data = np.arange(10000).reshape(100,100)
        peaks = np.arange(20).reshape(self.atoms_N,2)
        sublattice = Sublattice(
                peaks,
                image_data)
        sublattice.original_image = image_data
        for atom in sublattice.atom_list:
            atom.sigma_x = 2.
            atom.sigma_y = 2.
            atom.amplitude_gaussian = 10.
            atom.amplitude_max_intensity = 10.
        self.sublattice = sublattice

    def test_plot_amplitude_sigma_hist2d(self):
        plot_amplitude_sigma_hist2d(self.sublattice)

    def test_plot_atom_column_hist_amplitude_gauss2d_maps(self):
        plot_atom_column_hist_amplitude_gauss2d_maps(self.sublattice)

    def test_plot_atom_column_histogram_sigma(self):
        plot_atom_column_histogram_sigma(self.sublattice)

    def test_plot_atom_column_histogram_amplitude_gauss2d(self):
        plot_atom_column_histogram_amplitude_gauss2d(self.sublattice)

    def test_plot_atom_column_histogram_max_intensity(self):
        plot_atom_column_histogram_max_intensity(self.sublattice)

    def test_plot_amplitude_sigma_scatter(self):
        plot_amplitude_sigma_scatter(self.sublattice)

    def test_get_atom_list_atom_sigma_range(self):
        atom_list = get_atom_list_atom_sigma_range(self.sublattice, (1., 3.))
        self.assertEqual(len(atom_list), self.atoms_N)
