"""Read-only CPU audit of the completed GAT/GATv2 checkpoints and ledgers."""
import argparse
import json
from pathlib import Path

import torch

from run_attention_operator_pair import OperatorClassifier, ROOT, read, sha, state_digest
from summarize_attention_operator_pair import LABELS, analyze, check_history, require


def audit(source, archived_summary):
    summary = read(source / 'summary.json')
    require(summary == read(archived_summary), 'Server summary differs from the archived summary')
    require(read(source / 'status.json') == {'status': 'completed', 'exit_code': 0},
            'Server run is incomplete')
    analysis = analyze(summary)
    require(read(source / 'analysis.json') == analysis, 'Server analysis differs from recomputed metrics')
    results = {}
    for name, metrics in summary['models'].items():
        folder = source / name
        require(read(folder / 'metrics.json') == metrics, 'Arm metrics differ from summary: ' + name)
        ledger = folder / 'epochs.jsonl'
        history = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
        check_history(history, metrics, epochs=150, steps=211, samples=13446)
        validation_by_epoch = {0: metrics['epoch_zero']}
        validation_by_epoch.update({row['epoch']: row['validation'] for row in history})
        checkpoints = {
            'epoch_zero.pt': (0, 'epoch_zero'),
            'trained_best.pt': (metrics['best_trained_epoch'], 'trained_best_validation'),
            'best.pt': (metrics['best_epoch'], 'best_validation'),
            'last.pt': (150, 'last_validation'),
        }
        model = OperatorClassifier(name, seed=summary['protocol']['seed'])
        hashes = {}
        for filename, (epoch, phase) in checkpoints.items():
            path = folder / filename
            digest = sha(path)
            require(digest == metrics['checkpoint_sha256'][filename], 'Checkpoint SHA256 differs: ' + str(path))
            # Deserialize only after verifying bytes against the archived digest.
            payload = torch.load(path, map_location='cpu', weights_only=False)
            expected_config = {
                'variant': name,
                'protocol': summary['protocol'],
                'preparation_protocol': summary['preparation_protocol'],
                'binding': summary['binding'],
                'labels': LABELS,
                'initial_state_sha256': metrics['initial_state_sha256'],
            }
            require(payload['epoch'] == epoch and payload['config'] == expected_config
                    and payload['validation'] == metrics[phase] == validation_by_epoch[epoch],
                    'Checkpoint epoch, configuration or validation differs: ' + str(path))
            model.load_state_dict(payload['model_state'], strict=True)
            if epoch == 0:
                require(state_digest(model.state_dict()) == metrics['initial_state_sha256'],
                        'Epoch-zero state hash differs: ' + name)
            hashes[filename] = digest
        results[name] = {
            'checkpoint_sha256': hashes,
            'strict_cpu_state_load': True,
            'ledger_sha256': sha(ledger),
            'ledger_epochs': len(history),
            'optimizer_updates_in_ledger': sum(row['optimizer_updates'] for row in history),
            'best_epoch': metrics['best_epoch'],
            'best_macro_f1': metrics['best_validation']['macro_f1'],
        }
    return {
        'status': 'passed',
        'summary_sha256': sha(source / 'summary.json'),
        'archived_summary_sha256': sha(archived_summary),
        'paired_order_and_initial_rng_digests_match': True,
        'models': results,
        'raw_graph_replay_executed': False,
        'new_optimizer_updates': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archived-summary', type=Path,
                        default=ROOT / 'docs/attention_operator_pair_server_summary_20261105.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(audit(args.source, args.archived_summary), indent=2))
