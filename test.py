import unittest
import Numba_Box_Count_Stats as countoscope

# Countoscope test suite
# at the moment this doesn't do much except check no exceptions are thrown,
# but I hope it can grow in scope
# I have not included test_data.dat in the repository for the moment, let me (Adam) know if you need it

class TestCountoscope(unittest.TestCase):
    def test_squares_single_sep(self):
        countoscope.Calc_and_Output_Stats('test_data.dat', 2400, window_size_x=217.6, window_size_y=174,
                                          box_sizes=(2, 4, 8), sep_sizes=2)
        
    def test_squares_variable_sep(self):
        countoscope.Calc_and_Output_Stats('test_data.dat', 2400, window_size_x=217.6, window_size_y=174,
                                          box_sizes=(2, 4, 8), sep_sizes=(1, 2, 3))
        
    def test_rectangles_single_x(self):
        countoscope.Calc_and_Output_Stats('test_data.dat', 2400, window_size_x=217.6, window_size_y=174,
                                          box_sizes_x=(2, 4, 8), box_sizes_y=2, sep_sizes=2)
        
    def test_rectangles_single_y(self):
        countoscope.Calc_and_Output_Stats('test_data.dat', 2400, window_size_x=217.6, window_size_y=174,
                                          box_sizes_x=(2, 4, 8), box_sizes_y=2, sep_sizes=2)
        
    def test_rectangles_xy(self):
        countoscope.Calc_and_Output_Stats('test_data.dat', 2400, window_size_x=217.6, window_size_y=174,
                                          box_sizes_x=(2, 4, 8), box_sizes_y=(4, 8, 16), sep_sizes=2)

if __name__ == '__main__':
    unittest.main()