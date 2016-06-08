import unittest
from atom_position_class import Atom_Position

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
