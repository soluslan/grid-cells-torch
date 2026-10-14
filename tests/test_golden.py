import numpy as np
import pytest

import golden_cases


@pytest.fixture(scope="module")
def stored():
    with np.load(golden_cases.GOLDEN_PATH) as npz:
        return {k: npz[k] for k in npz.files}


@pytest.fixture(scope="module")
def current():
    return golden_cases.compute_golden()


def test_outputs_match_pre_restructure_golden(stored, current):
    for key, value in current.items():
        np.testing.assert_array_equal(value, stored[key], err_msg=key)


def test_shard0_unchanged(stored):
    digest = golden_cases.shard0_digest()
    if digest is None:
        pytest.skip("data/shards/0000-of-0099.npz not present")
    assert digest == str(stored["shard0_digest"])
