import atomap.api as am
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patheffects as patheffects

atom_lattice = am.dummy_data.get_fantasite_atom_lattice()

####
im = atom_lattice.image0
s_adf = hs.signals.Signal2D(im)
####

sublattice_B = atom_lattice.sublattice_list[0]
sublattice_B.construct_zone_axes()
zone = sublattice_B.zones_axis_average_distances[0]
s_dd = sublattice_B.get_atom_distance_difference_map([zone])
s_dd = s_dd.isig[40.:460.,40.:460.]
s_adf = s_adf.isig[40.:460.,40.:460.]

z1 = sublattice_B.zones_axis_average_distances[0]
z2 = sublattice_B.zones_axis_average_distances[1]
plane = sublattice_B.atom_planes_by_zone_vector[z2][23]
s_dd_line = sublattice_B.get_atom_distance_difference_line_profile(z1,plane)

###
fig = plt.figure(figsize=(4.3, 2)) # in inches
gs = gridspec.GridSpec(1, 5)

ax_adf = plt.subplot(gs[:2])
ax_al = plt.subplot(gs[2:4])
ax_lp = plt.subplot(gs[4])

cax_adf = ax_adf.imshow(
	np.rot90(s_adf.data),
	interpolation='nearest',
	origin='upper',
	extent=s_adf.axes_manager.signal_extent
)

fontprops = fm.FontProperties(size=12)
scalebar0 = AnchoredSizeBar(
        ax_adf.transData,
        10, '10 nm', 4,
        pad=0.1,
        color='white',
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops)
ax_adf.add_artist(scalebar0)

atoms_A = atom_lattice.sublattice_list[0]
for idx, x in enumerate(atoms_A.x_position):
    y = atoms_A.y_position[idx]
    if (240 < x < 350) and (96 < y < 200):
        ax_adf.scatter(y, x, color='r', s=0.5)

atoms_B = atom_lattice.sublattice_list[1]
for idx, x in enumerate(atoms_B.x_position):
    y = atoms_B.y_position[idx]
    if (240 < x < 350) and (96 < y < 200):
        ax_adf.scatter(y, x, color='b', s=0.5)

cax_al = ax_al.imshow(
	np.rot90(s_dd.data),
	interpolation='nearest',
	origin='upper',
	extent=s_adf.axes_manager.signal_extent,
	cmap='viridis'
)

scalebar1 = AnchoredSizeBar(
        ax_al.transData,
        10, '10 nm', 4,
        pad=0.1,
        color='white',
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops)
ax_al.add_artist(scalebar1)

for ax in [ax_adf, ax_al]:
    ax.set_xticks([])
    ax.set_yticks([])

ax_lp.plot(s_dd_line.data, s_dd_line.axes_manager[0].axis)
ax_lp.set_xlabel("Distance difference", fontsize=8)
ax_lp.set_ylabel(r"Distance from interface",
        fontsize=8)
ax_lp.tick_params(axis='both', which='major', labelsize=6)
ax_lp.tick_params(axis='both', which='minor', labelsize=6)
ax_lp.yaxis.set_label_position('right')
ax_lp.yaxis.set_ticks_position('right')
ax_lp.set_ylim(-103, 317)


path_effects = [patheffects.withStroke(linewidth=2, foreground='black', capstyle="round")]
ax_adf.text(
        0.015,0.90,"a",fontsize=12, color='white',
        path_effects=path_effects,
        transform=ax_adf.transAxes)
ax_al.text(
        0.015,0.90,"b",fontsize=12, color='white',
        path_effects=path_effects,
        transform=ax_al.transAxes)
ax_lp.text(
        0.05,0.90,"c",fontsize=12, color='w',
        path_effects=path_effects,
        transform=ax_lp.transAxes)

gs.update(left=0.01, wspace=0.05, top=0.95, bottom=0.2, right=0.89)
plt.savefig('Atom_lattice.png', dpi=300)
