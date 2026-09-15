import unittest
import numpy as np
from src.regression import make_folds, cross_validate, fit_ols, predict, prepare, analyze, predictors


class RegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.x = rng.normal(size=(100, 6))
        self.y = 350 + self.x @ np.arange(1, 7) + rng.normal(size=100)
        self.folds = make_folds(100)

    def test_folds_deterministic_balanced(self):
        np.testing.assert_array_equal(self.folds, make_folds(100))
        self.assertEqual(set(self.folds), set(range(10)))
        self.assertTrue(np.all(np.bincount(self.folds) == 10))
        self.assertFalse(np.array_equal(self.folds, make_folds(100, seed=7)))

    def test_training_only_scaling_and_outcomes(self):
        predicted, audit = cross_validate(self.x, self.y, self.folds)
        held = self.folds == 0
        changed_x = self.x.copy()
        changed_x[held] += 10000
        changed_y = self.y.copy()
        changed_y[held] += 10000
        _, changed_audit = cross_validate(changed_x, changed_y, self.folds)
        np.testing.assert_allclose(audit[0]['training_mean'], self.x[~held].mean(axis=0))
        np.testing.assert_allclose(audit[0]['training_mean'], changed_audit[0]['training_mean'])
        np.testing.assert_allclose(audit[0]['training_scale'], changed_audit[0]['training_scale'])
        changed_predictions, _ = cross_validate(self.x, changed_y, self.folds)
        np.testing.assert_allclose(predicted[held], changed_predictions[held])
        # Independently reproduce every fold using only the other nine folds.
        for fold in range(10):
            test = self.folds == fold
            fit = fit_ols(self.x[~test], self.y[~test])
            np.testing.assert_allclose(predicted[test], predict(fit, self.x[test]))
        self.assertEqual(sum(a['test_n'] for a in audit), len(self.y))
        self.assertTrue(np.isfinite(predicted).all())

    def test_residual_identity_unique_rows_and_heldout_metrics(self):
        names = predictors('gpt2')
        rows = [dict(item=1, zone=i+3, word='word', mean_RT=y,
                     **dict(zip(names, x))) for i, (x, y) in enumerate(zip(self.x, self.y))]
        output, summary, _, _ = analyze(rows, 'gpt2', self.folds)
        self.assertEqual(len(output), len({(r['item'], r['zone']) for r in output}))
        for r in output:
            self.assertEqual(r['residual'], r['observed_rt']-r['predicted_rt'])
        residuals = np.array([r['residual'] for r in output])
        self.assertAlmostEqual(summary['full']['rmse'], np.sqrt(np.mean(residuals**2)))

    def test_boundary_exclusions_and_missing_frequency(self):
        base = dict(word='word', mean_RT=300, **{c: 1 for c in predictors('gpt2')})
        rows = [dict(base, item=item, zone=zone) for item in (1, 2) for zone in (1, 2, 3)]
        for r in rows:
            if r['zone'] == 1:
                r.update(gpt2_surprisal='nan', prev_gpt2_surprisal='',
                         prev_word_length='', prev_log_frequency='')
            if r['zone'] == 2:
                r['prev_gpt2_surprisal'] = 'nan'
        rows[-1]['log_frequency'] = ''
        good, excluded, reasons = prepare(rows, 'gpt2')
        self.assertEqual([(r['item'], r['zone']) for r in good], [(1, 3)])
        self.assertEqual(len(excluded), 5)
        self.assertEqual(reasons['story_initial_surprisal'], 2)
        self.assertEqual(reasons['story_boundary_previous_predictors'], 2)
        self.assertEqual(reasons['previous_story_initial_surprisal'], 2)
        self.assertEqual(reasons['current_frequency_unavailable'], 1)


if __name__ == '__main__':
    unittest.main()
