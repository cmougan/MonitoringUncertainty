import unittest
import pytest

import numpy as np
import pandas as pd

from distributions import DistributionShift


class TestDistributionShift(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), columns=["a", "b", "c"]
        )
        self.X_array = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        self.y = np.array([0, 1.0, 2.0])

    def test_strategy_assert(self):
        with pytest.raises(ValueError):
            DistributionShift(strategy="aa")

    def test_param_float(self):
        with pytest.raises(ValueError):
            DistributionShift(param="aa")

    def test_covariateShift(self):

        # Param = 1
        db = DistributionShift(param=1.0)
        db.fit(self.X, self.y)
        np.testing.assert_equal(db.transform(self.X).mean(axis=0), np.array([3, 4, 5]))

        # Param = 2
        db = DistributionShift(param=2.0)
        db.fit(self.X, self.y)
        np.testing.assert_equal(db.transform(self.X).mean(axis=0), np.array([4, 5, 6]))

        # Param = 10
        db = DistributionShift(param=10.0)
        db.fit(self.X, self.y)
        np.testing.assert_equal(
            db.transform(self.X).mean(axis=0), np.array([12, 13, 14])
        )

        # On transform
        db = DistributionShift(param=1.0)
        db.fit(self.X, self.y)
        np.testing.assert_equal(
            db.transform(self.X, parameter=1).mean(axis=0), np.array([3, 4, 5])
        )
        np.testing.assert_equal(
            db.transform(self.X, parameter=2).mean(axis=0), np.array([4, 5, 6])
        )
        np.testing.assert_equal(
            db.transform(self.X, parameter=10).mean(axis=0), np.array([12, 13, 14])
        )

        # Param = 0
        db = DistributionShift(param=0.0)
        db.fit(self.X, self.y)
        np.testing.assert_equal(db.transform(self.X).mean(axis=0), np.array([2, 3, 4]))

    def test_handling_array(self):
        # Param = 1
        db = DistributionShift(param=1.0)
        db.fit(self.X_array, self.y)
        np.testing.assert_equal(
            db.transform(self.X_array).mean(axis=0), np.array([3, 4, 5])
        )

        # Param = 2
        db = DistributionShift(param=2.0)
        db.fit(self.X_array, self.y)
        np.testing.assert_equal(
            db.transform(self.X_array).mean(axis=0), np.array([4, 5, 6])
        )

        # Param = 10
        db = DistributionShift(param=10.0)
        db.fit(self.X_array, self.y)
        np.testing.assert_equal(
            db.transform(self.X_array).mean(axis=0), np.array([12, 13, 14])
        )

        # On transform
        db = DistributionShift(param=1.0)
        db.fit(self.X_array, self.y)
        np.testing.assert_equal(
            db.transform(self.X_array, parameter=1).mean(axis=0), np.array([3, 4, 5])
        )
        np.testing.assert_equal(
            db.transform(self.X_array, parameter=2).mean(axis=0), np.array([4, 5, 6])
        )
        np.testing.assert_equal(
            db.transform(self.X_array, parameter=10).mean(axis=0),
            np.array([12, 13, 14]),
        )

        # Param = 0
        db = DistributionShift(param=0.0)
        db.fit(self.X_array, self.y)
        np.testing.assert_equal(
            db.transform(self.X_array).mean(axis=0), np.array([2, 3, 4])
        )

    def test_does_nothing(self):
        db = DistributionShift()
        db.fit(self.X, self.y)
        np.testing.assert_equal(db.transform(self.X).mean(axis=0), np.array([2, 3, 4]))

    def test_col_selection(self):
        db = DistributionShift(cols=['a'],param=1)

        db.fit(self.X, self.y)
        np.testing.assert_equal(db.transform(self.X).mean(axis=0), np.array([3, 3, 4]))
