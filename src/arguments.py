"""!@file arguments.py

@brief Module for parsing command line arguments for all scripts in the project
"""

import argparse


def parse_cases(args):
    """!
    @brief Parse the 'cases' argument which specify the cases to predict on

    @details If 'all' is specified, then all cases are used.
             Otherwise, a list of case numbers can be specified (eg. '0,3,5' for Case_000, Case_003, Case_005)
    """
    if args.cases == "all":
        args.cases = [f"Case_{i:03}" for i in range(12)]
    else:
        case_nums = args.cases.split(",")  # split the string into a list of case numbers
        args.cases = []
        for num in case_nums:
            num = int(num)
            assert num >= 0 and num < 11, "Invalid case number"
            args.cases.append(f"Case_{num:03}")  # append the case number to the list of cases

    return args


def parse_args():
    """!
    @brief Parse command line arguments for all scripts in the project
    """
    parser = argparse.ArgumentParser()

    # for multiple scripts
    parser.add_argument("--dataroot", type=str, help="root directory of LCTSC dataset")
    parser.add_argument("--output_dir", type=str, help="output directory")

    # specific to train.py
    parser.add_argument(
        "--include_testing",
        type=int,
        help="0 or 1 - (Only used for train.py) specifies whether to evaluate the model \
            on the test set after each epoch of the training process",
    )

    # specific to predict.py
    parser.add_argument("--model_state_dict", type=str, help="path to the trained model state dict file")
    parser.add_argument(
        "--cases",
        default="all",
        type=str,
        help="'all' or specific case numbers eg. 0,3,5 for Case_000, Case_003, Case_005",
    )
    parser.add_argument("--prediction_type", default="prob", type=str, help="'prob' for probabilities or 'mask' for binary masks. MUST BE PROBABILITIES FOR THE ANALYSIS.")

    # specific to make_plots.py
    parser.add_argument("--predictions_dir", type=str, help="directory containing the predictions for each case")
    parser.add_argument("--metric_logger", type=str, help="file containing metric_logger")

    args = parser.parse_known_args()[0]
    args = parse_cases(args)

    return args
