import matplotlib
matplotlib.use('Agg')
import unittest
from atomap.atom_position import Atom_Position
from numpy import pi

class test_create_atom_position_object(unittest.TestCase):
    
#    def setUp(self):
#        self.test_image = np.arange(256*256).reshape(256,256)
#    
#    def tearDown(self):
#        self.detector_process.stop_running()

    def test_create_atom_position_object(self):
        atom_x = 10
        atom_y = 20
        atom_position = Atom_Position(atom_x, atom_y)
        self.assertEqual(atom_position.pixel_x, 10)
        self.assertEqual(atom_position.pixel_y, 20)

class test_atom_position_object_tools(unittest.TestCase):
    
    def setUp(self):
        self.atom_position = Atom_Position(1,1)
    
    def test_get_atom_angle(self):
        atom_position0 = Atom_Position(1,2)
        atom_position1 = Atom_Position(3,1)
        atom_position2 = Atom_Position(1,0)
        atom_position3 = Atom_Position(5,1)
        atom_position4 = Atom_Position(2,2)

        angle90 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position1)
        angle180 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position2)
        angle0 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position3)
        angle45 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position4)

        self.assertAlmostEqual(angle90, pi/2, 7)
        self.assertAlmostEqual(angle180, pi, 7)
        self.assertAlmostEqual(angle0, 0, 7)
        self.assertAlmostEqual(angle45, pi/4, 7)
