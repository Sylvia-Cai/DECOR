import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import eval_baselines as cli
from workflows import baseline_scores as baseline
from workflows import imt_scores as imt


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'scores.json'
        self.truth = {('Economy', str(i), 'test'): {
            'thought': 1 - i % 2, 'response': i % 2,
        } for i in range(20)}

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding='utf-8')
        return self.path

    def test_binary_response_ground_truth_and_mean(self):
        path = self.write([{'topic': 'Economy', 'question': str(i), 'results': {
            'test': {'eval': {'thought': 'decept',
                              'response': 'decept' if i else 'honest'}}
        }} for i in range(2)])
        result = baseline.evaluate_file_response_gt(path, self.truth)
        self.assertEqual(tuple(result), imt.CONDITIONS)
        self.assertEqual(result['response']['y_true'], [0, 1])
        self.assertEqual(result['combined']['y_score'], [0.5, 1.0])
        self.assertEqual(result['combined']['y_pred'], [1, 1])
        for paths in ([path], [path, path]):
            summary = cli.build_summary('binary', {
                'model': baseline.evaluate_model_response_gt(paths, self.truth)})
            self.assertIn('  Response\n', summary)
            self.assertIn('  Thought & Response\n', summary)
            self.assertNotIn('  Thought\n', summary)

    def test_normalized_scores_keep_ranking(self):
        path = self.write({'results': [{'topic': 'Economy', 'question': str(i),
            'condition': 'test', 'thought_normalized': score,
            'response_normalized': score} for i, score in enumerate([0.1, 0.4])]})
        result = baseline.evaluate_model_response_gt([path], self.truth)['single']
        self.assertEqual(result['response']['auroc'], 1.0)
        self.assertEqual(result['combined']['auroc'], 1.0)

    def test_cot_scale_does_not_depend_on_observed_max(self):
        path = self.write({'results': [{'topic': 'Economy', 'question': str(i),
            'condition': 'test', 'thought_raw': score,
            'response_raw': score} for i, score in enumerate([0.0, 1.0])]})
        result = cli._cot_evaluate_file_response_gt(path, self.truth)
        self.assertEqual(result['response']['auroc'], 1.0)
        self.assertEqual(result['response']['acc'], 0.5)
        self.assertNotIn('thought', result)

    def test_missing_thought_keeps_response(self):
        path = self.write([{'topic': 'Economy', 'question': '0', 'results': {
            'test': {'eval': {'response': 'honest'}}}}])
        result = baseline.evaluate_file_response_gt(path, self.truth)
        self.assertEqual(result['response']['y_true'], [0])
        self.assertEqual(result['combined']['y_true'], [])
        path = self.write({'results': [{'topic': 'Economy', 'question': '0',
            'condition': 'test', 'response_raw': 1.0}]})
        result = cli._cot_evaluate_file_response_gt(path, self.truth)
        self.assertEqual(result['response']['n_total'], 1)
        self.assertEqual(result['combined']['n_total'], 0)

    def test_scheming_uses_response_truth_and_arithmetic_mean(self):
        path = self.write([{'topic': 'Economy', 'question': str(i), 'results': {
            'test': {'scheming_eval': {'thought_score': ts, 'response_score': rs}}
        }} for i, (ts, rs) in enumerate([(10, 2), (1, 8)])])
        result = cli._scheming_evaluate_file_response_gt(path, self.truth)
        self.assertEqual(result['response']['auroc'], 1.0)
        self.assertEqual(result['combined']['auroc'], 0.0)
        self.assertNotIn('thought', result)
        aggregated = {'n_runs': 2, **{c: cli._aggregate_runs([result[c], result[c]])
                                    for c in baseline.CONDITIONS}}
        self.assertIn('Thought & Response', cli.build_summary('scheming', {'m': aggregated}))

    def test_eq4_all_units(self):
        units = [{'iu_id': 1, 'sis': 1}, {'iu_id': 2, 'sis': 3}]
        assessments = [{'iu_id': i, **{d: {'score': score} for d in imt.IMT_DIMS}}
                       for i, score in [(1, 1.0), (2, 0.0)]]
        self.assertEqual(imt.RESPONSE_STRATEGY, 'full_avg')
        self.assertEqual(imt.pool_score(assessments, units, imt.RESPONSE_STRATEGY), 0.25)

    def test_cv_empty_and_populated_schemas_match(self):
        records = [{'topic': 'Economy', 'label_response': i % 2,
                    'score_thought': {'full_avg': i % 2},
                    'score_response': {'full_avg': i % 2}} for i in range(20)]
        populated = imt.compute_binary_cv_response_gt(records)
        empty = imt.compute_binary_cv_response_gt([])
        self.assertEqual(set(empty['response']), set(populated['response']))
        self.assertEqual(populated['response']['f1_mean'], 1.0)
        aggregate = imt.aggregate_binary_response_gt_runs([empty, populated])
        self.assertEqual(aggregate['mean']['response']['f1'], 1.0)

    def test_missing_results_directory(self):
        self.assertEqual(baseline.list_baseline_families(Path(self.tmp.name) / 'missing'), [])


if __name__ == '__main__':
    unittest.main()
