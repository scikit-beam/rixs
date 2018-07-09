import pytest
import numpy as np
from rixs import preliminary_processing as pre_proc


@pytest.fixture
def data():
    # generate a list of 2D numpy arrays.
    out_list = []
    for i in range(0, 3):
        out_list.append(np.arange(1, 26).reshape(5, 5))

    return out_list


@pytest.fixture
def regions():
    # generate a 'regions' dictionary.
    out_dict = {}
    for i in range(0, 3):
        out_dict['region_{}'.format(i)] = [1, 3, 0, 2]

    return out_dict


@pytest.fixture
def expected_extracted_regions():
    # generate the expected output dictionary for extract_regions
    out_dict = {}
    for i in range(0, 3):
        out_dict['region_{}'.format(i)] = [np.array([[6,  7], [11, 12]]),
                                           np.array([[6,  7], [11, 12]]),
                                           np.array([[6,  7], [11, 12]])]

    return out_dict


def test_extract_regions(data, regions, expected_extracted_regions):
    # test extract_regions.
    for region in regions:
        expecteds = expected_extracted_regions[region]
        for i in range(0, len(expecteds)):
            expected = expecteds[i]
            actual = pre_proc.extract_regions(data, regions)[region][i]
            assert np.allclose(actual, expected)
