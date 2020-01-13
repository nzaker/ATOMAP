import os
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import atomap.api as am
import atomap.atom_finding_refining as afr

my_path = os.path.join(os.path.dirname(__file__), 'make_quantifying_scanning_distortions')
if not os.path.exists(my_path):
    os.makedirs(my_path)

np.random.seed(0)

##########################
# Get data
sublattice = am.dummy_data.get_scanning_distortion_sublattice()

# Plotting sublattice
signal = sublattice.signal
signal.plot()
signal._plot.signal_plot.figure.savefig(os.path.join(my_path, 'distortion_signal.png'))

##########################
# Getting cropped atom
atom_i = 200
atom = sublattice.atom_list[atom_i]
atom_x, atom_y = atom.pixel_x, atom.pixel_y
radius = 6
atom_image = atom._get_image_slice_around_atom(sublattice.image, radius)[0]
atom_mask = afr._make_mask_circle_centre(atom_image, radius)
atom_image[atom_mask] = 0

# Getting line center of mass
line_x_com_list = []
for ix in range(2, atom_image.shape[1] - 2):
    line_mask_x = atom_mask[:, ix]
    com_x_offset = line_mask_x[:round(len(line_mask_x)/2)].sum()
    line_x = atom_image[:, ix][np.invert(line_mask_x)]
    line_x_com = center_of_mass(line_x)[0] + com_x_offset
    line_x_com_list.append(line_x_com)

line_x_com_range = range(len(line_x_com_list))
line_x_com_poly = np.polyfit(line_x_com_range, line_x_com_list, deg=1)
line_x_com_fit = np.poly1d(line_x_com_poly)(line_x_com_range)
line_x_variation = np.array(line_x_com_list) - np.array(line_x_com_fit)

line_y_com_list = []
for iy in range(2, atom_image.shape[0] - 2):
    line_mask_y = atom_mask[iy]
    com_y_offset = line_mask_y[:round(len(line_mask_y)/2)].sum()
    line_y = atom_image[iy][np.invert(line_mask_y)]
    line_y_com = center_of_mass(line_y)[0] + com_y_offset
    line_y_com_list.append(line_y_com)

line_y_com_range = range(len(line_y_com_list))
line_y_com_poly = np.polyfit(line_y_com_range, line_y_com_list, deg=1)
line_y_com_fit = np.poly1d(line_y_com_poly)(line_y_com_range)
line_y_variation = np.array(line_y_com_list) - np.array(line_y_com_fit)

# Plotting figure
fig = plt.figure(figsize=(5, 5))
gs = plt.GridSpec(300, 200)
ax_com_x_image = fig.add_subplot(gs[0:100, 0:95])
ax_com_y_image = fig.add_subplot(gs[0:100, 105:200])
ax_com_y_line = fig.add_subplot(gs[118:190, :])
ax_com_y_line_variation = fig.add_subplot(gs[225:300, :])

cax_min = atom_image[np.invert(atom_mask)].min() * 0.95
cax_max = atom_image[np.invert(atom_mask)].max() * 1.05

# Image CoM x
cax_com_x_image = ax_com_x_image.imshow(atom_image)
cax_com_x_image.set_clim(cax_min, cax_max)
ax_com_x_image.scatter(range(2, atom_image.shape[0] - 2), line_x_com_list, s=1.27)

# Image CoM y
cax_com_y_image = ax_com_y_image.imshow(atom_image)
cax_com_y_image.set_clim(cax_min, cax_max)
ax_com_y_image.scatter(line_y_com_list, range(2, atom_image.shape[0] - 2), s=1.27)

# Line CoM y
ax_com_y_line.plot(line_y_com_range, line_y_com_list, label="Experimental")
ax_com_y_line.plot(line_y_com_range, line_y_com_fit, label="Line fit")
ax_com_y_line_variation.plot(line_y_com_range, line_y_variation)

# Subplot configurations
ax_com_x_image.set_axis_off()
ax_com_y_image.set_axis_off()

# Subplot annotations
ax_com_x_image.set_title("CoM x (slow scan)")
ax_com_y_image.set_title("CoM y (fast scan)")
ax_com_y_line.set_title("CoM y (fast scan)")
ax_com_y_line_variation.set_title(
        "CoM y (fast scan), line fit removed. Std: {0}".format(round(np.std(line_y_variation), 3)))

ax_com_y_line.legend()

# Saving figure
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
fig.savefig(os.path.join(my_path, 'explain_atom_shape.png'), dpi=200)

##########################
s_x, s_y, avg_x, avg_y = sublattice.estimate_local_scanning_distortion()
s_x.plot()
s_x._plot.signal_plot.figure.savefig(os.path.join(my_path, 'distortion_x_signal.png'))
