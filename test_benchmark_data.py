"""Dataset adapters, reproducible splits, provenance and per-dataset reporting."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_data import (SOURCES, SOURCE_SHA256, build_subsets, fingerprint, fixed_sample,
                            normalize_records, parse_reference, read_snapshot)


def fixture(source, count):
    if source.startswith('gsm8k'):
        rows = [{'question': f'{source} question {i}?', 'answer': f'Explain calculations.\n#### {i}'}
                for i in range(count)]
    else:
        rows = [{'ID': f'chal-{i}', 'Body': f'SVAMP body {i}.', 'Question': 'How many?',
                 'Answer': i, 'Equation': f'({i}+0)'} for i in range(count)]
    return normalize_records(source, rows)


class DatasetTests(unittest.TestCase):
    def test_reference_parser_requires_numeric_answer(self):
        self.assertEqual(parse_reference('Reasoning\n#### $1,234.50'), 1234.5)
        self.assertEqual(parse_reference(-7), -7)
        for value in (True, 'The answer is 21', '1/2', 'nan'):
            with self.assertRaises(ValueError):
                parse_reference(value)

    def test_svamp_body_and_question_are_both_present(self):
        row = normalize_records('svamp', [dict(ID='chal-1', Body='Pack costs 76.',
                                               Question='After discount 25?', Answer=51)])[0]
        self.assertEqual(row['question'], 'Pack costs 76. After discount 25?')
        self.assertEqual(row['expected'], 51)
        self.assertEqual(row['source_id'], 'svamp:challenge:chal-1')
        self.assertNotIn('Equation', row)

    def test_duplicate_source_id_is_rejected(self):
        row = dict(ID='same', Body='Pack costs 76.', Question='How much?', Answer=76)
        with self.assertRaises(ValueError):
            normalize_records('svamp', [row, row])

    def test_fixed_sampling_is_independent_of_input_order(self):
        rows = fixture('gsm8k_test', 20)
        a = fixed_sample(rows, 5, 203)
        self.assertEqual(a, fixed_sample(list(reversed(rows)), 5, 203))
        self.assertNotEqual(a, fixed_sample(rows, 5, 204))

    def test_sampling_does_not_depend_on_answers(self):
        rows = fixture('gsm8k_test', 20)
        altered = [dict(row, expected=-999) for row in rows]
        self.assertEqual([r['source_id'] for r in fixed_sample(rows, 5, 203)],
                         [r['source_id'] for r in fixed_sample(altered, 5, 203)])

    def test_exact_duplicates_and_exclusions(self):
        rows = fixture('gsm8k_test', 4)
        rows.append(dict(rows[0], source_id='duplicate'))
        selected = fixed_sample(rows, 3, 203, [rows[1]['question_sha256']])
        self.assertEqual(len({r['question_sha256'] for r in selected}), 3)
        self.assertNotIn(rows[1]['question_sha256'], {r['question_sha256'] for r in selected})
        with self.assertRaises(ValueError):
            fixed_sample(rows, 5, 203)

    def test_development_and_evaluation_are_disjoint(self):
        catalog = {name: fixture(name, 8) for name in SOURCES}
        # Simulate a train/test overlap and ensure it is excluded from development.
        catalog['gsm8k_train'][0] = dict(catalog['gsm8k_train'][0],
            question=catalog['gsm8k_test'][0]['question'],
            question_sha256=catalog['gsm8k_test'][0]['question_sha256'])
        subsets = build_subsets(catalog, dev_count=3, gsm_count=4, svamp_count=2)
        dev = {r['question_sha256'] for r in subsets['development']}
        evaluation = {r['question_sha256'] for r in subsets['evaluation']}
        self.assertFalse(dev & evaluation)
        self.assertFalse(dev & {r['question_sha256'] for r in catalog['gsm8k_test']})
        self.assertEqual(len(subsets['evaluation']), 6)
        self.assertTrue(all(r['source_split'] == 'train' for r in subsets['pilot']))

    def test_snapshot_tampering_is_detected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/'gsm8k_test.jsonl'
            raw = b'{"question":"A?", "answer":"#### 21"}\n'
            path.write_bytes(raw)
            metadata = dict(SOURCES['gsm8k_test'], sha256=hashlib.sha256(raw).hexdigest(), rows=1)
            (Path(directory)/'gsm8k_test.metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            spec = dict(SOURCES['gsm8k_test'], count=1)
            with patch.dict(SOURCES, {'gsm8k_test': spec}), patch.dict(
                    SOURCE_SHA256, {'gsm8k_test': hashlib.sha256(raw).hexdigest()}):
                rows, _ = read_snapshot('gsm8k_test', directory, download=False)
                self.assertEqual(rows[0]['expected'], 21)
                path.write_bytes(raw.replace(b'21', b'22'))
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    read_snapshot('gsm8k_test', directory, download=False)

    def test_offline_mode_never_silently_downloads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with self.assertRaises(FileNotFoundError):
                read_snapshot('svamp', directory, download=False)

    def test_reports_keep_datasets_and_call_references_separate(self):
        from plot_results import generate_report_and_plots
        rows = []
        for dataset, tot_calls in [('gsm8k', 10), ('svamp', 20)]:
            for method, calls in [('Tree-of-Thoughts (LLM-Judge)', tot_calls),
                                  ('Cost-Aware Priority Search', 5)]:
                rows.append(dict(dataset=dataset, method=method, seed=11, problem_id=0,
                                 ans=21, correct=True, calls=calls, tokens=10, sec=1))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/'input.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            summary = generate_report_and_plots(directory, [path])
            cost = summary[summary.method.eq('Cost-Aware Priority Search')].set_index('dataset')
            self.assertEqual(cost.loc['gsm8k', 'Call_Savings_vs_ToT_%'], 50)
            self.assertEqual(cost.loc['svamp', 'Call_Savings_vs_ToT_%'], 75)
            self.assertEqual(len(summary), 4)


if __name__ == '__main__':
    unittest.main()
