import unittest

from evaluation.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_at_k(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"b", "d"}, 3), 0.5)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "b", "a"], {"a", "b"}), 0.5)

    def test_no_relevant_result(self):
        self.assertEqual(reciprocal_rank(["x", "y"], {"z"}), 0.0)

    def test_mean_reciprocal_rank(self):
        runs = [(["a", "b"], {"b"}), (["x", "y"], {"z"})]
        self.assertAlmostEqual(mean_reciprocal_rank(runs), 0.25)


if __name__ == "__main__":
    unittest.main()
