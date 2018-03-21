import os
import atomap.api as am
import atomap.dummy_data as dummy_data
import hyperspy.api as hs


my_path = os.path.dirname(__file__) + '/integrate/'
if not os.path.exists(my_path):
    os.makedirs(my_path)


def voronoi_integration(sublattice):
    s = dummy_data.get_simple_cubic_signal(image_noise=True)
    s_integrated = am.integrate(
            s.data, sublattice.x_position, sublattice.y_position)
    s2 = hs.signals.Signal2D(s_integrated[2])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig('Voronoi1.png', overwrite=True)
    s2 = hs.signals.Signal2D(s_integrated[1])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig('Voronoi2.png', overwrite=True)


def watershed_integration(sublattice):
    s = dummy_data.get_simple_cubic_signal(image_noise=True)
    s_integrated = am.integrate(
            s.data, sublattice.x_position, sublattice.y_position,
            method='Watershed')
    s2 = hs.signals.Signal2D(s_integrated[2])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig('Watershed1.png', overwrite=True)
    s2 = hs.signals.Signal2D(s_integrated[1])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig('Watershed2.png', overwrite=True)


s = dummy_data.get_simple_cubic_signal(image_noise=True)

atom_positions = am.get_atom_positions(s, separation=7)
sublattice = am.Sublattice(atom_positions, image=s.data)
sublattice.find_nearest_neighbors()
sublattice.refine_atom_positions_using_2d_gaussian()

voronoi_integration(sublattice)
watershed_integration(sublattice)
