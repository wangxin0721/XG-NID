# Cross-layer CPU fixture RNG scope repair

The user-provided server output for a0a867a records six standard-library
checks passing and all five tensor cases failing at the same models()
fixture assertion. None reached its individual functional checks. The
launcher stopped before metadata validation, output/log creation, CUDA
acceptance or training.

The fixture previously snapshotted the CPU RNG before deep-copying three
PyG-backed inputs for each variant. Its assertion therefore measured both
PyG copy reconstruction and the CrossLayerClassifier constructor. PyG copy
reconstruction can consume RNG, so that measurement does not isolate the
new projection constructor's RNG preservation.

The repaired fixture prepares independent encoder/head copies first, then
snapshots RNG immediately before constructing each variant. It asserts
unchanged RNG separately for CL-Last and CL-Multi, with the variant in the
failure message. The production build() already measures construction after
preparing the backbone/head. No RNG assertion or model acceptance check is
removed, and both variants' initialization and parameter checks remain.

Model, protocol, training runner, report generator and launcher remain byte
identical to a0a867a. Both arms still run all150 epochs. The repaired five
tensor cases must run on the server: local torch/PyG are unavailable. Six
standard checks and five Python3.9 syntax checks pass locally.

Fetch the repair commit, archive scripts/configs/src/xgnid and the reference
docs/packet_pool_results_20261102.json into a fresh temporary directory, then
run launch_cross_layer_pair.sh from the server repository. The previous
test failure created no formal run artifacts, so no output deletion or
recovery is required. The optional torch-spline-conv warning is separate
from the assertion that stopped this launch.
