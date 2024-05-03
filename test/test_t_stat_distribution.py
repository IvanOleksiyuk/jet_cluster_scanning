from t_statistic_distribution import t_statistic_distribution
import unittest
import numpy as np

class TestTStatisticsDistribution(unittest.TestCase):
	"""Class for testing cs_performance"""

	def setUp(self) -> None:
		pass

	def test_t_statistic_distribution(self):
		"""Test evluation of maxdev5 method"""
		results = t_statistic_distribution(
			["test/config/prep05_1_maxdev3_msdeCURTAINS_1mean.yaml",
			"test/config/bootstrap_sig_contam_ideal.yaml",
			"test/config/plot_path2.yaml",
			"test/config/small.yaml"]
		)	
		np.testing.assert_almost_equal(results["TS_list"][:10], [5.79210748, 3.88921048, 4.6750361 , 0.0, 4.16855121, 5.03201769, 3.05049478, 4.93289818, 3.92147318, 4.35384848])
		np.testing.assert_almost_equal(results["ps"][0][:4], [0.5, 0.16666666666666666, 0.0, 0.5])