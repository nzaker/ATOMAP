import os
import numpy as np
import atomap.api as am

my_path = os.path.join(os.path.dirname(__file__), 'makeexamplesimages')
if not os.path.exists(my_path):
    os.makedirs(my_path)

#####
s_dumbbell = am.dummy_data.get_dumbbell_heterostructure_signal()
s_dumbbell.plot()
s_dumbbell._plot.signal_plot.figure.savefig(os.path.join(my_path, 'dumbbell_heterostructure.png'))

#####
s_precipitate = am.dummy_data.get_precipitate_signal()
s_precipitate.plot()
s_precipitate._plot.signal_plot.figure.savefig(os.path.join(my_path, 'precipitate.png'))

#####
s_perovskite = am.dummy_data.get_fantasite()
s_perovskite.plot()
s_perovskite._plot.signal_plot.figure.savefig(os.path.join(my_path, 'perovskite.png'))
