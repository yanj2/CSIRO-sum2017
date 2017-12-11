"""
Random sampling can become significantly biased if the size of the data select
is not large enough. Something to be wary of.

Methods that can be employed to deal with this issue include:
- Stratified sampling: Population is divided into homogeneous subgroups called
strata, and the right percentage of instances are sampled from each stratum
to gaurantee that the test set is representative of the overall population. 
"""

import numpy as np
import get_data
import hashlib
# train_test_split can take in multiple datasets with the same number of test_indices
# and split across those indices. Useful if labels are kept separately, or if
# you have multiple datasets
from sklearn.model_selection import train_test_split


def split_train_test(data, test_ratio):
    """
    function taken from textbook, divides the data into a training set and
    a test set using random selection. NOTE: Running the program multiple times
    will give different results each time. This means that its possible that then
    ML program will have seen the whole data set already.
    """

    # original
    shuffled_indices = np.random.permutation(len(data))

    # solution 1: However these solutions will both break with updated datasets
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def hash_split(data, test_ratio, id_column, hash=hashlib.md5):

    """
    possible implementation for hash solution. Where you select the data from
    the set based off its hash value
    hash function: RSA's md5, internet RFC 1321
    """

    #NOTE: we want to use the most stable features to build unique identifier.
    # Stable referring to somethign that won't change much in the future.
    # No deletions etc.
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))

    # `~` is a bitwise operation for flipping the bits. binary ones complement
    return data.loc[~in_test_set], data.loc[in_test_set]

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

if __name__ == "__main__":

    housing_with_id = get_data.load_housing_data().reset_index()     # adds an index column
    train_set, test_set = hash_split(housing_with_id, 0.2, "id")

    quit()
    train_set, test_set = split_train_test(get_data.load_housing_data(), 0.2)
