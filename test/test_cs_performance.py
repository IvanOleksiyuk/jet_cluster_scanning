import configure_tests
import unittest
import numpy as np
from cs_performance_evaluation import cs_performance_evaluation
import sys


class TestCSPerformance(unittest.TestCase):
    """Class for testing cs_performance"""

    def setUp(self) -> None:
        """Download a test data"""
        self.counts_windows = np.load(sys.path[0] + "/test_materials/count_windows.npy")
        self.binning = np.load(sys.path[0] + "/test_materials/binning.npy")

    def test_maxdev5(self):
        """Test evluation of maxdev5 method"""
        res = cs_performance_evaluation(
            counts_windows=self.counts_windows,
            binning=self.binning,
            config_file_path=sys.path[0] + "/config/testCSE_maxdev5.yaml",
        )
        self.assertAlmostEqual(3.508741407313602, res)

    # def test_2meansder(self):
    #     """Test evluation of 2meansder method"""
    #     res = cs_performance_evaluation(
    #         counts_windows=self.counts_windows,
    #         binning=self.binning,
    #         config_file_path=sys.path[0] + "/config/testCSE_2meansder.yaml",
    #     )
    #     self.assertAlmostEqual(3.5990916567517726, res)
    #     pass

    def test_maxdev5_with_plotting(self):
        """Test evluation of >5sigma method"""
        res = cs_performance_evaluation(
            counts_windows=self.counts_windows,
            binning=self.binning,
            config_file_path=[
                sys.path[0] + "/config/testCSE_maxdev5.yaml",
                sys.path[0] + "/config/testCSE_plotting.yaml",
            ],
        )


if __name__ == "__main__":
    TT = TestCSPerformance()
    TT.setUp()
    TT.test_maxdev5_with_plotting()
