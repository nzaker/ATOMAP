import os
import matplotlib.pyplot as plt
import atomap.api as am

my_path = os.path.join(os.path.dirname(__file__), 'automation')
if not os.path.exists(my_path):
    os.makedirs(my_path)


s = am.dummy_data.get_two_sublattice_signal()
process_parameter = am.process_parameters.PerovskiteOxide110()
atom_lattice = am.make_atom_lattice_from_image(
        s, process_parameter=process_parameter, pixel_separation=14)

atom_lattice.plot()
plt.gcf().savefig(os.path.join(my_path, 'atom_lattice.png'))
