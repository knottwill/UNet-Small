import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # for multiple scripts
    parser.add_argument("--dataroot", type=str, help="root directory of LCTSC dataset")
    parser.add_argument("--output_dir", type=str, help="output directory (to save predictions in)")

    # specific to train.py
    parser.add_argument(
        "--include_testing",
        type=int,
        help="0 or 1 - whether to evaluate the model \
                        on the test set at each epoch of the training process",
    )

    args = parser.parse_known_args()[0]
    return args
