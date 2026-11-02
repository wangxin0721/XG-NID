# One P0/P1 training-budget confirmation

This is the single additional paired run requested on 2026-11-02 to check
whether stopping at epoch20 affected the earlier packet-pooling conclusion.
It introduces no new architecture, learning rate, seed, split, or search.
The completed `packet_pool_pair_s42_recovery_v2` results remain unchanged.

Both variants restart from the same D32-Gate epoch62 checkpoint
(`7815875a111d3e94ceefd0e9ef6cb64dedaf6ff80723645f1ec035b376af31ff`,
Macro-F1 97.32070542814468%). They do not resume the old epoch20 states or
Adam optimizers. P0 uses the existing ordered mean readout; P1 uses the same
64/16/1 attention network as the completed screen. Trainable parameters
remain 7144 and 8200, respectively. Encoders, BN parameters, and statistics
stay frozen/eval.

## Registered budget and selection

- Exactly one pair, P0 then P1, each completing150 epochs (31650 updates).
- Early stopping is disabled. Registered min50 and patience20 do not truncate
  either variant; stale counters describe trained-epoch progress only.
- Seed42, fixed split, train/eval512/16, existing head LR1e-5, P1 pool LR1e-4,
  fresh Adam with weight decay0, original final-output NLL.
- Save `epoch_zero.pt`, `trained_best.pt`, `best.pt`, and `last.pt` separately.
  Select trained-best only from epochs1..150. Final-best compares it with
  epoch0 and retains epoch0 on ties.
- Reset RNG and optimizer independently; verify all150 epoch sample orders
  and the starting RNG match across variants.

The dedicated entry reuses the fixed-budget training loop in
`run_relation_pair.py` with packet-specific minibatching, optimization,
architecture, and evaluation functions in a separate function namespace.
It does not mutate that runner or the historical packet runner. Historical
protocols and source files remain byte-identical.

## Frozen cache reuse and acceptance

Copy only the train/val frozen node caches from the completed screen.
Validate its completed status, original protocol, mature source binding,
source-code hashes, both20-epoch histories, checkpoints/prediction hashes,
cache metadata, encoder/split identities, sample order, and file/tensor hashes.
Preserve and rehash every file in the completed source before and after the
confirmation. No failed-run recovery or training-state resume is involved.

CPU tensor suites run first, followed by metadata validation. During the
submitted job, the unchanged CUDA repeat/uniform/backward preflight runs
before cache loading. Initial pair/source identity, original-graph replays,
gradient probes, clean-state/RNG restoration, and frozen-encoder checks remain
mandatory. Matched readouts use atol/rtol1e-6, raw/cache comparison1e-5, and
legacy source predictions and confusion matrices must match exactly.

Replay both the selected-best and trained-best checkpoints on all original
validation graphs; share a replay only when their epoch and model state are
identical. Account for each real graph-load phase in the parent process.
Cache-extraction reads from the old run are provenance only. Memory estimates
describe the original extraction, not current copy/loading or process RSS.

## Interpretation

Compare the first20 validation scores, sample-order hashes, and train NLL
with the original histories. Record exact matching flags and the largest
NLL difference. If any differs, report that the first20 trajectory was not
exactly reproduced and do not attribute the result solely to removing early
stopping. The new run still reports its own first20-versus-full150 gain.

Report epoch0, trained-best, selected-best, last, trained-best epoch, and
per-class metrics. Preserve changed sample IDs and prediction transitions,
with attention to WebBased/Recon and Spoofing/BruteForce. Compute
`P1-max(P0,D32)` with the same +0.10 percentage-point budget threshold, and
record whether that verdict changed after epoch20. This threshold is not a
statistical-significance claim. No automatic extra round, seed, model, or
hyperparameter experiment follows this confirmation. No test set is evaluated.

## Server launch

Archive the published commit's `scripts configs src/xgnid` and
`docs/packet_pool_results_20261102.json` into a new temporary directory. The
JSON record is required by the standard-library checks. From the server
repository run:

```bash
bash "$bundle/scripts/launch_packet_pool_confirmation.sh" 1
tail -n 50 -F outputs/packet_pool_confirmation_s42_v1.log
```

The GPU argument is the physical GPU index. The launcher refuses an existing
output/log/PID instead of silently overwriting or starting another round.
Outputs are separate from the relationship-aggregation experiment:
`outputs/packet_pool_confirmation_s42_v1`. Return `status.json`, `report.md`,
`analysis.json`, and `packet_diagnostics.json` for interpretation.
