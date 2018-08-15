import numpy as np
import matplotlib.pyplot as plt
import atomap.initial_position_finding as ipf


class TestAddAtomAdderRemoving:

    def test_no_atoms_input(self):
        data = np.random.random((200, 200))
        peaks = ipf.add_atoms_with_gui(data, [])
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((100, 100))
        assert len(peaks) == 0
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 1
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 0

    def test_atoms_input(self):
        data = np.random.random((200, 50))
        peaks = ipf.add_atoms_with_gui(data, [[120., 20.], [120., 30.], ])
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((100, 10))
        assert len(peaks) == 2
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 3
        fig.canvas.button_press_event(120., 20., 1)
        assert len(peaks) == 2
