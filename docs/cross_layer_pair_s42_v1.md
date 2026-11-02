# CL-Last / CL-Multi: frozen-depth readout comparison

Only the Packet cross-layer readout changes. Both variants start from the
original D32-Gate epoch62 (97.32070542814468%); neither uses the P1 nor
Rel-Gate candidate weights. The original GAT relations, message aggregation,
BN parameters/statistics, and Flow readout remain frozen/eval.

Capture Packet layer1 after its original BN and LeakyReLU, and layer2 after
the corresponding BN/activation. Pool each independently to64 dimensions
using the existing ordered mean backend. CL-Last feeds `[h2,h2]` into a
new Linear(128,64); CL-Multi feeds `[h1,h2]`. Both initialize the left block
to zero, right block to identity, bias to zero. There is no new activation
or normalization. The new projection remains outside no_grad and precedes
the unchanged D32 64-to32 projection, dynamic fusion gate and classifier.

Both add8256 parameters and train15400 parameters total. The duplicate
input control matches the interface and parameter count; its two weight
blocks are redundant and do not add depth information. This experiment
tests complementary shallow information, not a layer-to-layer residual.

## Fixed protocol

- Seed42 and original split; fresh Adam, WD0, original final NLL.
- Existing fusion/projection/head LR1e-5; new projection LR1e-4 in both arms.
- Train/eval batch512/16, workers2, exactly150 epochs and31650 updates each.
- Early stopping disabled. min50/patience20 are retained metadata only.
- Epoch0 saved and eligible; trained-best selected independently from1..150;
  final-best is max(epoch0,trained-best), preferring epoch0 on ties.
- Independent optimizer/RNG reset and all150 sample-order hashes matched.
- Save epoch_zero.pt, trained_best.pt, best.pt, last.pt and separate
  trained-best predictions. Report per-class metrics and changed sample IDs.

## Cache and acceptance

Build a fresh graph-level cache containing flow64, Packet first64/second64,
legacy mean for diagnostics, labels, ordered IDs, split and encoder binding.
Do not reuse last-layer-only pooling caches or relation caches. Report
concatenation tensor-memory estimates including retained caches, explicitly
excluding Python/model/graph memory and allocator overhead.

CPU tests exercise real GAT encoders, initial identity, post-activation
capture, empty relations, parameter count/RNG preservation, depth dependence,
raw/cache reordering, strict reload, cache rejection, hook cleanup, gradients,
and clean-state recovery after successful and failed probes.
CUDA preflight checks initial equality and nonzero finite gradients through
both64-column projection blocks without optimizer updates or RNG changes.
Raw-graph three-update probes verify both blocks and all original downstream
groups receive gradients and change, while encoders/BN remain unchanged.
Finally restore clean weights, gradients and RNG and discard probe Adam.

Paired and matched-source initialization uses atol/rtol1e-6 and exact source
predictions/confusion matrix. Independent legacy CUDA probability errors are
recorded separately. Raw/cache replay uses1e-5 and exact predictions. Replay
selected-best and trained-best on all13446 original validation graphs;
share a replay only if their epoch and model state are equal. Count worker
loads via parent-verified stage receipts, not worker-private counters.

## Decision and server use

`delta=F1(CL-Multi)-max(F1(CL-Last),F1(D32))`. The expansion threshold stays
+0.10 percentage points. Also report comparison against P1's
97.42719120553645% (verified from user-provided matrices, without rehashing
the checkpoint). P1 is a score reference only, not the matched control or
an additional mandatory threshold. No significance claim, test-set evaluation,
residual, pooling/relationship combination, extra seed or automatic follow-up.

Archive the published commit's `scripts configs src/xgnid` plus
`docs/packet_pool_results_20261102.json` (needed by standard tests). From
the server repository run `bash "$bundle/scripts/launch_cross_layer_pair.sh" 1`.
The launcher runs six standard checks, five CPU tensor checks and metadata
validation before submitting the GPU job. It refuses existing output/log/PID.
The job runs CUDA and real-graph acceptance before training. Outputs live
under `outputs/cross_layer_pair_s42_v1`; return status.json, report.md,
analysis.json and cross_layer_diagnostics.json.
