import configure_tests
import unittest
import numpy as np
from cluster_scanning import ClusterScanning
import sys


class TestClusterScanning(unittest.TestCase):
    """Class for testing cs_performance"""

    def test_run_full(self):
        cs = ClusterScanning("test/config/testCS.yaml")
        cs.run()
        self.assertAlmostEqual(cs.counts_windows_sum[0, 1], 12358)

    def test_run_fast(self):
        cs = ClusterScanning("test/config/testCSfast.yaml")
        cs.run()


if __name__ == "__main__":
    TCS = TestClusterScanning()
    TCS.test_run()
