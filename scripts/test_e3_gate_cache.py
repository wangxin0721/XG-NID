"""Pure arithmetic/contract checks; no models, graphs or synthetic model run."""
import unittest

from analyze_e3_gate_cache import failure_bounds, failure_category, prediction


class CacheLogicChecks(unittest.TestCase):
    def test_real_report_count_bounds_do_not_invent_joint_counts(self):
        self.assertEqual(failure_bounds(196, 182, 140), {"minimum": 42, "maximum": 56})
        self.assertEqual(failure_bounds(246, 246, 245), {"minimum": 1, "maximum": 1})
        with self.assertRaises(ValueError):
            failure_bounds(196, 197, 140)

    def test_failure_partitions_distinguish_direction_and_fusion(self):
        self.assertEqual(failure_category(False, False, True, [.2, .8]), "correct_branch_favored_but_fusion_wrong")
        self.assertEqual(failure_category(False, False, True, [.8, .2]), "wrong_branch_favored_and_fusion_wrong")
        self.assertEqual(failure_category(False, False, True, [.5, .5]), "tied_weights_with_one_correct_branch_and_fusion_wrong")
        self.assertEqual(failure_category(True, False, False, [.2, .8]), "recovered_when_both_single_interventions_wrong")

    def test_cached_probabilities_reject_nonfinite_and_wrong_shape(self):
        with self.assertRaises(ValueError):
            prediction([1., 0.])
        with self.assertRaises(ValueError):
            prediction([float("nan")] + [0.] * 7)
        with self.assertRaises(ValueError):
            prediction([.2] * 8)


if __name__ == "__main__":
    unittest.main()
