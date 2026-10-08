import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("deepmind_lab") is not None:
        return
    skip = pytest.mark.skip(reason="deepmind_lab not installed")
    for item in items:
        if "dmlab" in item.keywords:
            item.add_marker(skip)
