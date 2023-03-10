# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import graph_from_file, kruskal
import unittest   # The test framework

class Test_MST(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file("input/network.1.in")
        g_mst = kruskal(g)
        min_power_with_path_expected = (9, [7, 14, 1, 8, 4, 12, 9])
        self.assertEqual(g_mst.min_power1(7,9),  min_power_with_path_expected)

if __name__ == '__main__':
    unittest.main()