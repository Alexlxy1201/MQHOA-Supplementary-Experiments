import unittest
import numpy as np

from mqhoa_supp.statistics import bootstrap_mean, holm, paired_analysis


class StatisticsTests(unittest.TestCase):
    def test_all_equal_differences(self):
        result=paired_analysis([0]*6,np.random.default_rng(1),100)
        self.assertEqual(result["primary_p"],1)
        self.assertEqual(result["ci95_low"],0)
        self.assertEqual(result["rank_biserial"],0)

    def test_exact_two_sided_sign_flip(self):
        result=paired_analysis([1,2,3,4,5,6],np.random.default_rng(1),100)
        self.assertEqual(result["primary_p"],2/64)
        self.assertEqual(result["rank_biserial"],1)

    def test_holm(self):
        rows=[{"primary_p":.04},{"primary_p":.001},{"primary_p":.03}]
        holm(rows)
        np.testing.assert_allclose([r["holm_p"] for r in rows],[.06,.003,.06])

    def test_bootstrap_reproducible(self):
        a=bootstrap_mean([1,2,3,4],np.random.default_rng(5),100)
        b=bootstrap_mean([1,2,3,4],np.random.default_rng(5),100)
        self.assertEqual(a,b)
        self.assertEqual(bootstrap_mean([7],np.random.default_rng(1)),(7,7))


if __name__=="__main__":
    unittest.main()
