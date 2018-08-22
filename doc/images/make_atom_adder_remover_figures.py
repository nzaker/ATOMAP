import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import atomap.api as am
import atomap.testing_tools as tt
import atomap.tools as at

my_path = os.path.join(os.path.dirname(__file__), 'atomadderremovergui')
if not os.path.exists(my_path):
    os.makedirs(my_path)

#####
s = am.dummy_data.get_distorted_cubic_signal()
peaks = am.get_atom_positions(s, 25)
peaks_new = am.add_atoms_with_gui(s, peaks)
ifig = plt.get_fignums()[0]
fig = plt.figure(ifig)
ax = fig.axes[0]

position_list = [
        [-20, 20],
        [51, 215],
        [90, 215],
        [129, 215],
        [159, 220],
        [170, 215],
        [159, 220],
        [211.5, 215],
        [232, 250],
        ]
frames = at._generate_frames_position_list(position_list, num=10)

fargs = [fig, ]
at._draw_cursor(ax, frames[0][0], frames[0][1])
anim = FuncAnimation(fig, at._update_frame, frames=frames, fargs=fargs, interval=200, repeat=False)
anim.save(os.path.join(my_path, "add_atoms.gif"), writer='imagemagick')
plt.close(fig)
