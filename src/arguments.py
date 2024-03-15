import argparse


def parse_cases(args):
    """Parse the 'cases' argument which specify the cases to predict on"""

    if args.cases == "train":
        args.cases = [f"Case_{i:03}" for i in range(0, 8)]
    elif args.cases == "test":
        args.cases = [f"Case_{i:03}" for i in range(8, 12)]
    elif args.cases == "all":
        args.cases = [f"Case_{i:03}" for i in range(0, 12)]
    else:
        case_nums = args.cases.split(",")
        args.cases = []
        for num in case_nums:
            num = int(num)
            assert num >= 0 and num < 11, "Invalid case number"
            args.cases.append(f"Case_{num:03}")

    return args


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

    # specific to predict.py
    parser.add_argument("--model_state_dict", type=str, help="path to the trained model state dict file")
    parser.add_argument(
        "--cases",
        default="test",
        type=str,
        help="train, test, all, or specific case numbers \
                        eg. 0,3,5 for Case_000, Case_003, Case_005",
    )
    parser.add_argument("--prediction_type", default="prob", type=str, help="'prob' for probabilities or 'mask' for binary masks")

    args = parser.parse_known_args()[0]
    args = parse_cases(args)

    return args
