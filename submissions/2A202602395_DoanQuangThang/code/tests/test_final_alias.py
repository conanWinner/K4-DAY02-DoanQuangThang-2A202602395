"""Regression checks for one test evaluation reused under two reporting names.

Small generated fixtures are retained; no files or directories are deleted.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepweeds_lab.final_alias import reuse_final_predictions


@dataclass
class FixtureConfig:
    seed: int = 0
    backbone: str = 'fixture'


class TestFinalAlias(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='deepweeds-alias-'))
        self.cfg = FixtureConfig()
        self.source = self.root / 'final_cache/F01/seed0'
        self.source.mkdir(parents=True)
        lock = {'config': asdict(self.cfg), 'method': 'fivecrop', 'T': 0.7}
        result = {'exp_id': 'F01', 'seed': 0, 'method': 'fivecrop', 'T': 0.7,
                  'macro_f1_val': 0.8, 'macro_f1_test': 0.79, 'top1_test': 0.8,
                  'ece_test': 0.01, 'p95_ms': 30, 'latency': {'p95': 30}}
        (self.source / 'locked_seed_config.json').write_text(json.dumps(lock))
        (self.source / 'result.json').write_text(json.dumps(result))
        (self.source / 'test_logits.npz').write_bytes(b'fixture cached logits')
        for split in ('val', 'test'):
            folder = self.root / 'predictions'; folder.mkdir(exist_ok=True)
            (folder / f'F01_seed0_{split}.csv').write_text('fixture predictions ' + split)
        (self.root / 'curves').mkdir()
        (self.root / 'curves/F01_seed0.png').write_bytes(b'fixture curve')

    def reuse(self, cfg=None, method='fivecrop'):
        return reuse_final_predictions(cfg or self.cfg, method, 'F01', 'R01', self.root)

    def test_reuse_is_idempotent_and_preserves_source(self):
        original = (self.source / 'result.json').read_bytes()
        result = self.reuse()
        self.assertTrue(result['test_forward_reused'])
        self.assertTrue(result['latency_reused'])
        self.assertEqual(result['reused_from'], 'F01')
        self.assertEqual(result['exp_id'], 'R01')
        self.assertEqual(self.reuse(), result)
        self.assertEqual((self.source / 'result.json').read_bytes(), original)
        self.assertEqual((self.root / 'predictions/R01_seed0_test.csv').read_bytes(),
                         (self.root / 'predictions/F01_seed0_test.csv').read_bytes())

    def test_different_method_is_rejected_before_copy(self):
        with self.assertRaises(ValueError): self.reuse(method='single')
        self.assertFalse((self.root / 'final_cache/R01').exists())

    def test_different_training_config_is_rejected_before_copy(self):
        with self.assertRaises(ValueError): self.reuse(cfg=replace(self.cfg, backbone='another'))
        self.assertFalse((self.root / 'final_cache/R01').exists())

    def test_conflicting_destination_is_preserved(self):
        target = self.root / 'predictions/R01_seed0_test.csv'
        target.write_text('existing different predictions')
        with self.assertRaises(ValueError): self.reuse()
        self.assertEqual(target.read_text(), 'existing different predictions')
        self.assertFalse((self.root / 'final_cache/R01').exists())

    def test_existing_historical_timing_is_not_rewritten(self):
        result = self.reuse()
        path = self.root / 'final_cache/R01/seed0/result.json'
        result.pop('reused_from'); result.pop('test_forward_reused'); result.pop('latency_reused')
        result['p95_ms'] = 35; result['latency'] = {'p95': 35}
        path.write_text(json.dumps(result)); original = path.read_bytes()
        self.assertEqual(self.reuse()['p95_ms'], 35)
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
