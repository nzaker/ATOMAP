from pytest import approx
import numpy as np
import matplotlib.pyplot as plt
import atomap.initial_position_finding as ipf


class TestAddAtomAdderRemoving:

    def test_no_atoms_input(self):
        data = np.random.random((200, 200))
        peaks = ipf.add_atoms_with_gui(data)
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((100, 100))
        assert len(peaks) == 0
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 1
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 0

    def test_atoms_input(self):
        data = np.random.random((50, 200))
        peaks = ipf.add_atoms_with_gui(data, [[120, 20], [120, 30], ])
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform((100, 10))
        assert len(peaks) == 2
        fig.canvas.button_press_event(x0, y0, 1)
        assert len(peaks) == 3
        x1, y1 = fig.axes[0].transData.transform((120, 30))
        fig.canvas.button_press_event(x1, y1, 1)
        assert len(peaks) == 2
        assert peaks[0] == approx([120, 20])
        assert peaks[1] == approx([100, 10])

    def test_many_atoms_input(self):
        data = np.random.random((500, 1000))
        x, y = np.mgrid[10:990:100j, 10:490:100j]
        x, y = x.flatten(), y.flatten()
        positions = np.stack((x, y)).T
        peaks = ipf.add_atoms_with_gui(data, positions)
        assert len(peaks) == len(positions)
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform((peaks[0][0], peaks[0][1]))
        fig.canvas.button_press_event(x0, y0, 1)
        assert len(peaks) == len(positions) - 1

    def test_distance_threshold(self):
        data = np.random.random((50, 200))
        peaks0 = ipf.add_atoms_with_gui(data, distance_threshold=10)
        fig0 = plt.figure(1)
        x00, y00 = fig0.axes[0].transData.transform((100, 10))
        fig0.canvas.button_press_event(x00, y00, 1)
        assert len(peaks0) == 1
        x01, y01 = fig0.axes[0].transData.transform((109, 19))
        fig0.canvas.button_press_event(x01, y01, 1)
        assert len(peaks0) == 0
        plt.close(fig0)

        peaks1 = ipf.add_atoms_with_gui(data, distance_threshold=4)
        fig1 = plt.figure(1)
        x10, y10 = fig1.axes[0].transData.transform((100, 10))
        fig1.canvas.button_press_event(x10, y10, 1)
        assert len(peaks1) == 1
        x11, y11 = fig1.axes[0].transData.transform((109, 19))
        fig1.canvas.button_press_event(x11, y11, 1)
        assert len(peaks1) == 2
        plt.close(fig1)
