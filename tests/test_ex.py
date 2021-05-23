
import unittest
import pytest
import numpy as np

import distributions
c = distributions.ensemble.StdForestClassifier()
class TestSuma(unittest.TestCase):
    def test_bueno(self):
        np.testing.assert_equal(2,1+1)

