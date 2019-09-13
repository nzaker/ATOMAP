import os
import atomap.api as am
import matplotlib.pyplot as plt

my_path = os.path.join(os.path.dirname(__file__), 'quant')
if not os.path.exists(my_path):
    os.makedirs(my_path)

tdata = am.dummy_data.get_atom_counting_data()
atom_positions = am.get_atom_positions(tdata, 8, threshold_rel=0.1)
sublattice = am.Sublattice(atom_positions, tdata.data)
sublattice.construct_zone_axes()
sublattice.refine_atom_positions_using_2d_gaussian()

models = am.quant.get_statistical_quant_criteria([sublattice], 10)

plt.savefig(os.path.join(my_path, 'criteria_plot.png'), overwrite=True)

atom_lattice = am.quant.statistical_quant(sublattice.image, sublattice, models[3], 4)

plt.savefig(os.path.join(my_path, 'quant_output1a.png'), overwrite=True)

atom_lattice.plot()

plt.savefig(os.path.join(my_path, 'quant_output1b.png'), overwrite=True)
