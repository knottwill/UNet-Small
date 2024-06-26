# Small U-Net for Lung Segmentation

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.2.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.12.2-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>

## Description

This project trains a small version of U-Net for lung segmentation on 12 cases from the Lung CT Segmentation Challenge (2017). The report for the study can be found in `report/`. All results in the report can be reproduced using the scripts in `scripts/`.

The model (and metric logger) I trained locally is saved in `Models/UNet_wdk24.pt` and `Models/metric_logger_wdk24.pkl`.

<details open>
<summary><b>Toggle Examples</b></summary>
<br>
  <img src="./plots/example.png" alt="U-Net Small architecture">
</details>
<br>
<details open>
<summary><b>Toggle Project Structure</b></summary>

- `Dataset/` - LCTSC dataset
- `docs/` - Documentation for the project
- `Models/` - Directory containing trained model state dictionaries and metric loggers of their training.
- `plots/` - Directory containing plots/visualisations used in the repot
- `Predictions/` - Directory containing model predictions (probabilities)
- `report/` - Contains project report
- `scripts/` - Main scripts for re-producing the results of the projects
- `src/` - Source code containing re-usuable components used in the scripts
- `tests/` - unit tests
- `.gitignore` - Tells git which files to ignore
- `.pre-commit-config.yaml` - Specifies pre-commit hooks to protect the `main` branch
- `Dockerfile` - Dockerfile to generate docker image
- `environment.yml` - Conda environment used for the project
- `LICENSE` - MIT license.
- `train_test_split.json` - File containing the train/test split used in the project
</details>

## Architecture

Schematic of the architecture of our U-Net small implementation:

![U-Net Small architecture](./plots/UNet-Small.png)

## Usage / Re-production

#### 1. Set-up

To re-create the environment used for the project, you can either use conda or docker. I HIGHLY recommend using conda for best performance and not docker, since the docker container will not naturally have access to the `mps` or `cuda` device. Running the `train.py` and `predict.py` scripts with a CPU will take far longer.

```bash
# Option 1: re-create conda environment
$ conda env create -f environment.yml -n <env-name>

# Option 2 (not recommended): Generate docker image and run container
$ docker build -t <image_name> .
$ docker run -ti <image_name>
```

Check everything is working by running `pytest`.

Re-production is done by running the scripts in the order given blow. Options for the scripts are specified by passing arguments. Use the `--help` argument on any of the scripts. The same argument parser is used for all scripts, hence many of the possible arguments will be redundant for different scripts. All commands below assume they are being run from the root directory of the project, and hence use relative paths. If this is not the case, or your dataset/predictions/models etc. are not in the default locations, adjust the arguments where necessary. More details can be found in the docstrings of each script.

#### 1. Pre-processing

First, generate summary statistics of the dataset and perform pre-processing as described in the report:

```bash
$ python scripts/dataset_summary.py --dataroot ./Dataset # Summary of dataset
$ python scripts/preprocessing.py --dataroot ./Dataset # Pre-process data

# --dataroot: root directory of LCTSC dataset
```

#### 2. Training

Next, train the model. This script will save the state dictionary as `UNet.pt` and metric logger as `metric_logger.pkl` in the `--output_dir`.

```bash
$ python ./scripts/train.py --dataroot ./Dataset --output_dir ./Models --include_testing 1

# --dataroot: The root directory of the LCTSC dataset
# --output_dir: The directory to save the trained model and the metric logger
# --include_testing: Whether to evaluate the model on the test set after each epoch (this is necessary to re-produce the results from the report)
```

#### 3. Prediction

Get predictions of the trained model for the whole dataset. DO NOT CHANGE `--prediction_type prob`.

```bash
$ python scripts/predict.py --dataroot ./Dataset --model_state_dict ./Models/UNet_wdk24.pt --output_dir ./Predictions --cases all --prediction_type prob

# --dataroot: The root directory of the LCTSC dataset
# --model_state_dict: The path to the trained model state dict file. To use your newly trained model, replace ./Models/UNet_wdk24.pt with your the new state dict file (by default, it would have been saved in `./Models/UNet.pt`)
# --output_dir: The directory to save the predictions.
# --cases: The cases to predict on. Can be 'all' or a list of case numbers (eg. '0,3,5')
# --prediction_type: The type of prediction to save ('prob' for probabilities or 'mask' for binary masks). MUST BE 'prob' TO RE-PRODUCE RESULTS IN REPORT (we plot a precision-recall curve)
```

#### 4. Analysis

Generate the main performance statistics, and plot visualisations of examples in the test set.

```bash
$ python scripts/stats_and_visualisation.py --dataroot ./Dataset --predictions_dir ./Predictions --output_dir ./plots

# --dataroot: The root directory of the dataset
# --predictions_dir: The directory containing the predictions made by the model (should be probabilities)
# --output_dir: The directory where the visualisations will be saved
```

Generate the various plots used in the report:

```bash
$ python scripts/make_plots.py --dataroot ./Dataset --predictions_dir ./Predictions --metric_logger ./Models/metric_logger_wdk24.pkl --output_dir ./plots

# --dataroot: The root directory of the LCTSC dataset
# --predictions_dir: The directory containing the predictions made by the model (should be probabilities)
# --metric_logger: The path to the metric logger pickle file. To use the metric logger of your newly trained model, replace ./Models/metric_logger_wdk24.pkl with  ./Models/metric_logger.pkl
# --output_dir: The directory where the plots will be saved
```

## Timing

Times to run each script:
- `dataset_summary.py` - 1 minute
- `preprocessing.py` - 1 minute
- `train.py` - 20 minutes (using `mps` device)
- `predict.py` - 3 minutes (using `mps` device)
- `stats_and_visualisation.py` - 1 minute
- `make_plots.py` - 5 minute

I ran all scripts on my personal laptop. The `train.py` and `predict.py` scripts both used the `mps` device, which is essentially the macbook GPU. The specifications are:
- Operating System: macOS Sonoma v14.0

CPU:
- Chip:	Apple M1 Pro
- Total Number of Cores: 8 (6 performance and 2 efficiency)
- Memory (RAM): 16 GB

GPU (`mps`):
- Chipset Model: Apple M1 Pro
- Type: GPU
- Bus: Built-In
- Total Number of Cores: 14
- Metal Support: Metal 3
