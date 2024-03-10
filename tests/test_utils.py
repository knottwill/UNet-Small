import os
import sys

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils import slice_num


def test_slice_num():
    assert slice_num("1-041.dcm") == 41
    assert slice_num("1-42.dcm") == 42
    assert slice_num("Case_005/1-003.dcm") == 3
