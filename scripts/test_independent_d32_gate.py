"""Check independent D32 initialization and intermediate archive inventory."""
from pathlib import Path
import tempfile
import unittest

import torch

import run_independent_d32_gate as run
from test_packet_transition_readout import fixture


class IndependentD32Checks(unittest.TestCase):
    def test_historical_initialization_recipe_preserves_n0_encoders(self):
        torch.set_num_threads(2)
        _, graphs = fixture()
        device = torch.device('cpu')
        warm = run.core.Batch.from_data_list(graphs)
        n0 = run.baseline.build_experiment('E2', warm)
        run.core.initialize(n0, 'E2', warm)
        before = run.raw.state_digest(n0.state_dict())
        self.assertEqual(run.streams_for_seed(42), run.dim.initialization_streams(42))
        heads = {seed: run.model_from_stage('D32-Gate', n0, device, seed) for seed in (43, 44)}
        self.assertNotEqual(*(run.raw.state_digest(item.fusion.state_dict()) for item in heads.values()))
        for model in heads.values():
            self.assertTrue(run.preserves_n0_encoders('D32-Gate', model, n0))
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 7144)
            with torch.no_grad():
                self.assertEqual(tuple(model(warm).shape), (len(graphs), 8))
                self.assertTrue(bool((model.last_flow_weights == 0.5).all()))
        c = run.model_from_stage('C', n0, device, 43)
        with torch.no_grad():
            c.packet_relations()[0].lin_src.weight.add_(1)
        self.assertTrue(run.preserves_n0_encoders('C', c, n0))
        self.assertFalse(run.preserves_n0_encoders('D32-Gate', c, n0))
        self.assertEqual(run.raw.state_digest(n0.state_dict()), before)

    def test_inventory_requires_own_n0_and_matching_protocol(self):
        settings = run.raw.read(run.baseline.CONFIG)['models']
        protocols = run.stage_protocols(run.raw.read(run.CONFIG))
        bound = settings['N043']['sha256']['best.pt']
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            def add(run_name, stage, binding, **overrides):
                folder = repo / 'outputs' / run_name / stage
                folder.mkdir(parents=True)
                for file in ('best.pt', 'metrics.json', 'validation_predictions.jsonl.gz'):
                    (folder / file).touch()
                run.raw.write(folder / 'config.json', {'protocol': {**protocols[stage], 'seed': 43, **overrides},
                                                       'binding': {'n0_best_sha256': binding}})

            add('matching', 'D32-Gate', bound)
            add('other-parent', 'C', settings['N044']['sha256']['best.pt'])
            add('different-budget', 'JCL-Multi', bound, epochs=149)
            rows = run.inventory(repo, settings, protocols)['seeds']['43']['candidates']
            self.assertEqual({row['stage']: row['status'] for row in rows}, {
                'D32-Gate': 'ready_for_replay', 'C': 'n0_lineage_unverified', 'JCL-Multi': 'protocol_differs'})


if __name__ == '__main__':
    unittest.main()
