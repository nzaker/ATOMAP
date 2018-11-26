from pytest import approx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import atomap.initial_position_finding as ipf
import atomap.tools as at
from atomap.sublattice import Sublattice
import atomap.testing_tools as tt


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


class TestToggleAtomRefinePosition:

    def test_toggle_one(self):
        atom_position_list = [[10, 10], [10, 20]]
        sublattice = Sublattice(atom_position_list, np.zeros((30, 30)))
        sublattice.toggle_atom_refine_position_with_gui()
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((10, 20))
        assert sublattice.atom_list[1].refine_position
        fig.canvas.button_press_event(x, y, 1)
        assert not sublattice.atom_list[1].refine_position
        fig.canvas.button_press_event(x, y, 1)
        assert sublattice.atom_list[1].refine_position

    def test_toggle_all(self):
        atom_position_list = [[10, 10], [10, 20]]
        sublattice = Sublattice(atom_position_list, np.zeros((30, 30)))
        sublattice.toggle_atom_refine_position_with_gui()
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform((10, 10))
        x1, y1 = fig.axes[0].transData.transform((10, 20))
        fig.canvas.button_press_event(x0, y0, 1)
        assert not sublattice.atom_list[0].refine_position
        fig.canvas.button_press_event(x1, y1, 1)
        assert not sublattice.atom_list[1].refine_position
        fig.canvas.button_press_event(x0, y0, 1)
        assert sublattice.atom_list[0].refine_position
        fig.canvas.button_press_event(x1, y1, 1)
        assert sublattice.atom_list[1].refine_position

    def test_with_fitting(self):
        x_pos, y_pos = [[10, 10], [10, 20]]
        delta_pos = 1
        test_data = tt.MakeTestData(30, 30)
        test_data.add_atom_list(x_pos, y_pos)
        sublattice = test_data.sublattice
        atom0, atom1 = sublattice.atom_list
        atom0.pixel_x += delta_pos
        atom0.pixel_y += delta_pos
        atom1.pixel_x += delta_pos
        atom1.pixel_y += delta_pos
        sublattice.toggle_atom_refine_position_with_gui()
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform(
                (x_pos[0] + delta_pos, y_pos[0] + delta_pos))
        fig.canvas.button_press_event(x0, y0, 1)
        print(atom0.refine_position)
        print(atom1.refine_position)
        sublattice.refine_atom_positions_using_center_of_mass(
                mask_radius=4)
        assert atom0.pixel_x == (x_pos[0] + delta_pos)
        assert atom0.pixel_y == (y_pos[0] + delta_pos)
        assert atom1.pixel_x != (x_pos[1] + delta_pos)
        assert atom1.pixel_y != (y_pos[1] + delta_pos)


class TestDrawCursor:

    def test_simple(self):
        fig0, ax0 = plt.subplots()
        ax0.imshow(np.arange(100).reshape(10, 10))
        fig0.canvas.draw()
        fig1, ax1 = plt.subplots()
        ax1.imshow(np.arange(100).reshape(10, 10))
        fig1.canvas.draw()
        assert fig0.canvas.tostring_rgb() == fig1.canvas.tostring_rgb()

        fig2, ax2 = plt.subplots()
        ax2.imshow(np.arange(100).reshape(10, 10))
        at._draw_cursor(ax2, 5, 8)
        fig2.canvas.draw()
        assert fig0.canvas.tostring_rgb() != fig2.canvas.tostring_rgb()

    def test_draw_outside(self):
        fig0, ax0 = plt.subplots()
        ax0.imshow(np.arange(100).reshape(10, 10))
        fig0.canvas.draw()
        fig1, ax1 = plt.subplots()
        ax1.imshow(np.arange(100).reshape(10, 10))
        at._draw_cursor(ax1, -1, 8)
        fig1.canvas.draw()
        assert fig0.canvas.tostring_rgb() != fig1.canvas.tostring_rgb()


class TestUpdateFrame:

    def test_simple(self):
        fig, ax = plt.subplots()
        ax.imshow(np.arange(100).reshape(10, 10))
        at._draw_cursor(ax, 5, 8)
        frames = [[5, 9, False], [2, 2, True]]
        fargs = [fig, ]
        FuncAnimation(fig, at._update_frame, frames=frames,
                      fargs=fargs, interval=200, repeat=False)
        plt.close(fig)


class TestGenerateFramesPositionList:

    def test_simple(self):
        position_list = [[10, 10], [30, 20]]
        frames = at._generate_frames_position_list(position_list, num=10)
        assert len(frames) == 12
        assert frames[0][0:2] == position_list[0]
        assert frames[-1][0:2] == position_list[-1]
        assert frames[0][-1] is True
        assert frames[-1][-1] is True
        for frame in frames[1:-1]:
            assert frame[-1] is False

    def test_num(self):
        position_list = [[10, 10], [30, 20]]
        frames = at._generate_frames_position_list(position_list, num=16)
        assert len(frames) == 18
        assert frames[0][0:2] == position_list[0]
        assert frames[-1][0:2] == position_list[-1]
        assert frames[0][-1] is True
        assert frames[-1][-1] is True
        for frame in frames[1:-1]:
            assert frame[-1] is False

    def test_longer(self):
        position_list = [[10, 10], [30, 20], [50, 30], [20, 54], [13, 89]]
        frames = at._generate_frames_position_list(position_list, num=10)
        assert len(frames) == 45
