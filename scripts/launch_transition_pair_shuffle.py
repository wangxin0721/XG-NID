"""Background entry: True/Shuffle, including automatic publication on exit/failure."""
import argparse
from pathlib import Path
import sys

import launch_joint_transition_fusion as launcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--probe-only', action='store_true')
    sys.exit(launcher.launch(parser.parse_args(), output_name='transition_pair_shuffle_s42_v1',
                             runner='run_transition_pair_shuffle.py'))


if __name__ == '__main__':
    main()
