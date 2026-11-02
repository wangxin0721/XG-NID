# Relation pair CPU fixture repair (2026-11-02)

The server launch of `bc1a8e3fa33c98a9244dcc902447cf8e953a97b6`
stopped in the CPU tensor suite: seven cases ran, with two errors. The
launcher had not created the formal run directory, log, or PID file and had
not started formal training.

## Repairs

- The all-empty Packet-to-Packet fixture left PyG's lazy edge projection
  uninitialized, so freezing its parameters failed. The fixture now warms
  the encoders on a separate batch containing real relation edges and checks
  that no uninitialized parameters remain before constructing the classifier.
  The actual empty-relation batch retains zero edges and zero edge attributes;
  the original bias-only output and absence of a new relation mask are still
  checked.
- The checkpoint round-trip test previously deep-copied a PyG model and then
  failed the frozen-encoder eval guard. It now constructs an independent
  model, strictly loads the state, explicitly enters eval mode, and checks
  every submodule's mode before comparing predictions. This follows the
  production graph-replay lifecycle.

Only test setup and validation documentation change. Production model,
runner, launcher, protocol, numerical tolerances, empty-relation handling,
and both variants' full 150-epoch budgets remain unchanged. Frozen encoder
and BN checks remain mandatory.

## Validation and retry

Nine standard-library checks and Python 3.9 syntax checks for five files
pass locally. Ten production and historical files were compared byte for
byte with `bc1a8e3` and remain identical. The local environment has neither
torch nor PyG, so the repaired seven-case tensor suite still requires server
execution. CUDA preflight, original-graph replay, and formal training have
not been run locally.

Fetch the repair commit, archive `scripts configs src/xgnid` to a fresh
temporary directory, and run its `scripts/launch_relation_pair.sh 1` from
the server repository. The launcher first reruns the CPU tests, then metadata
checks, and only then creates `outputs/relation_pair_s42_v1.log` and submits
training. The previous CPU-test failure needs no cache recovery or output
deletion. The reported optional `torch-spline-conv` import warning is not
the exception that stopped these tests.
