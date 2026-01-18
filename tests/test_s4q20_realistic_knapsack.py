# This will work if ran from the root folder.
import sys 
sys.path.append("src/delivery_network")

from graph import graph_from_file, kruskal
import unittest   # The test framework

class Test_Realistic_Knapsack(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file("data/network.1.in")
        g_mst = kruskal(g)
        g_mst.dfs14()
        alloc_expected = (556524.188273085, {0: 116, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0})
        self.assertEqual((g_mst.realistic_knapsack(25e9, 1, 1, 0.01, 0.001)[0], g_mst.realistic_knapsack(25e9, 1, 1, 0.01, 0.001)[1]), alloc_expected)

if __name__ == '__main__':
    unittest.main()
