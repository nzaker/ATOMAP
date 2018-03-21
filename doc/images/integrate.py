import os
import atomap.api as am
import hyperspy.api as hs


my_path = os.path.dirname(__file__) + '/integrate/'
if not os.path.exists(my_path):
    os.makedirs(my_path)


def voronoi_integration(signal, sublattice):
    s_integrated = am.integrate(
            signal, sublattice.x_position, sublattice.y_position)
    s2 = hs.signals.Signal2D(s_integrated[2])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig(
            my_path + 'Voronoi1.png', overwrite=True)
    s2 = hs.signals.Signal2D(s_integrated[1])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig(
            my_path + 'Voronoi2.png', overwrite=True)


def watershed_integration(signal, sublattice):
    s_integrated = am.integrate(
            signal, sublattice.x_position, sublattice.y_position,
            method='Watershed')
    s2 = hs.signals.Signal2D(s_integrated[2])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig(
            my_path + 'Watershed1.png', overwrite=True)
    s2 = hs.signals.Signal2D(s_integrated[1])
    s2.plot(cmap='viridis')
    s2._plot.signal_plot.figure.savefig(
            my_path + 'Watershed2.png', overwrite=True)


signal = am.dummy_data.get_simple_cubic_signal(image_noise=True)
sublattice = am.dummy_data.get_simple_cubic_sublattice(image_noise=True)

voronoi_integration(signal, sublattice)
watershed_integration(signal, sublattice)
