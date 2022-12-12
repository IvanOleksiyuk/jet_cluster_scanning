import configure_tests
import unittest
import numpy as np
from cs_performance_evaluation import cs_performance_evaluation
import sys


class TestCSPerformance(unittest.TestCase):
    """Class for testing cs_performance"""

    def setUp(self) -> None:
        """Download a test data"""
        self.counts_windows = np.load(
            sys.path[0] + "/test_materials/count_windows.npy"
        )

    def test_m5sigma(self):
        """Test evluation of >5sigma method"""
        res = cs_performance_evaluation(
            counts_windows=self.counts_windows,
            save_path="",
            filterr="med",
            plotting=False,
            labeling=">5sigma",
            verbous=False,
        )
        self.assertEqual(3.508741407313602, res["chisq_ndof"])

    def test_other(self):
        """Test evluation of  method"""
        res = cs_performance_evaluation(
            counts_windows=self.counts_windows,
            save_path="",
            filterr="med",
            plotting=False,
            labeling="kmeans_der",
            verbous=False,
        )
        self.assertEqual(3.5990916567517726, res["chisq_ndof"])
        pass


# counts_windows = np.load(sys.path[0] + "/test_materials/count_windows.npy")
# res = cs_performance_evaluation(
#     counts_windows=counts_windows,
#     save_path="",
#     filterr="med",
#     plotting=False,
#     labeling=">5sigma",
#     verbous=False,
# )
# print(res["chisq_ndof"])

# counts_windows = np.load(sys.path[0] + "/test_materials/count_windows.npy")
# res = cs_performance_evaluation(
#     counts_windows=counts_windows,
#     save_path="",
#     filterr="med",
#     plotting=False,
#     labeling="kmeans_der",
#     verbous=False,
# )
# print(res["chisq_ndof"])
