import doctest
import matplotlib
matplotlib.use('Agg')

import atomap.atom_finding_refining
doctest.testmod(atomap.atom_finding_refining)

import atomap.atom_lattice
doctest.testmod(atomap.atom_lattice)

import atomap.atom_plane
doctest.testmod(atomap.atom_plane)

import atomap.atom_position
doctest.testmod(atomap.atom_position)

import atomap.initial_position_finding
doctest.testmod(atomap.initial_position_finding)

import atomap.io
doctest.testmod(atomap.io)

import atomap.main
doctest.testmod(atomap.main)

import atomap.plotting
doctest.testmod(atomap.plotting)

import atomap.process_parameters
doctest.testmod(atomap.process_parameters)

import atomap.stats
doctest.testmod(atomap.stats)

import atomap.sublattice
doctest.testmod(atomap.sublattice)

import atomap.testing_tools
doctest.testmod(atomap.testing_tools)
