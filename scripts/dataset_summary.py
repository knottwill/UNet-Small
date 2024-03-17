"""
python scripts/dataset_summary.py --dataroot ./Dataset
"""

import os
from os.path import join
import sys
from pydicom import dcmread
import pandas as pd

# add project root to sys.path
project_root = os.path.abspath(join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.arguments import parse_args

args = parse_args()

print("++++++++++ Dataset Summary ++++++++++++\n")

cases = []
data = []
for dir, subdirs, filenames in os.walk(join(args.dataroot, "Images")):
    for filename in filenames:
        if filename.endswith(".dcm"):
            dcm = dcmread(join(dir, filename))
            case_id = os.path.basename(dir)

            if case_id in cases:
                continue

            cases.append(case_id)
            data.append(
                [
                    case_id,
                    dcm["Manufacturer"].value,
                    dcm["ManufacturerModelName"].value,
                    dcm["PatientSex"].value,
                    dcm["KVP"].value,
                    dcm["DistanceSourceToDetector"].value,
                    dcm["DistanceSourceToPatient"].value,
                    dcm["Exposure"].value,
                ]
            )


df = pd.DataFrame(data, columns=["Case_ID", "Manufacturer", "Manufacturer_Model", "Patient_S", "KVP", "Distance_Source_To_Detector", "Distance_Source_To_Patient", "Exposure"])

# sort by case_id
df = df.sort_values(by=["Case_ID"], ignore_index=True)

pd.set_option("display.max_columns", None)
print(df)
