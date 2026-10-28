"""Independent N0 entry: warm-start E2 with equal fusion, no frozen layers."""
from run_e2_concat_pair import main

if __name__ == '__main__':
    main(default_variant='N0', default_mode='train')
