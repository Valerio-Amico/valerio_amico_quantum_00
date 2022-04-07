import lib.functions0 as f0
import unittest

class tester(unittest.TestCase):

    def test_trotter_step(self):
        f0.trotter_step_matrix(0.1)