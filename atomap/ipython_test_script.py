from atomap_tools import get_peak2d_skimage
from atomap_atom_finding_refining import subtract_average_background, do_pca_on_signal, construct_zone_axes_from_atom_lattice
from sub_lattice_class import Atom_Lattice
from hyperspy.api import load
import numpy as np
from atomap_atom_finding_refining import refine_atom_lattice

s_adf_filename = "tests/datasets/test_ADF_cropped.hdf5"
peak_separation = 0.15

s_adf = load(s_adf_filename)
s_adf.change_dtype('float64')
s_adf_modified = subtract_average_background(s_adf)
s_adf_modified = do_pca_on_signal(s_adf_modified)
pixel_size = s_adf.axes_manager[0].scale
pixel_separation = peak_separation/pixel_size

peaks = get_peak2d_skimage(
        s_adf_modified, 
        pixel_separation)[0]

atom_lattice = Atom_Lattice(
        peaks, 
        np.rot90(np.fliplr(s_adf_modified.data)))
atom_lattice.original_adf_image = np.rot90(np.fliplr(s_adf.data))
atom_lattice.pixel_size = pixel_size

construct_zone_axes_from_atom_lattice(atom_lattice)
