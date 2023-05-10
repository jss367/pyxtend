import numpy as np
import pandas as pd
import tensorflow as tf

from src.pyxtend import struct


def test_string():
    res = struct("Hello, world!")
    assert res == "str"


def test_small_list():
    res = struct([1, 2, 3])
    assert res == {"list": ["int", "int", "int"]}


def test_empty_list():
    res = struct([])
    assert res == {"list": []}


def test_small_set():
    small_set = {1, "this"}
    res = struct(small_set)
    assert res == {"set": ["int", "str"]} or res == {"set": ["str", "int"]}


def test_large_set():
    large_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    res = struct(large_set)
    assert res == {"set": ["int", "int", "int", "...10 total"]}


def test_empty_range():
    res = struct(range(0))
    assert res == {"range": []}


def test_range():
    res = struct(range(10))
    assert res == {"range": ["int", "int", "int", "...10 total"]}


def test_long_list():
    long_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = struct(long_list)
    assert res == {"list": ["int", "int", "int", "...9 total"]}


def test_mixed_list():  # udpate this one
    mixed_list = [1, "this", "sam", [4, 3, "here"]]
    res = struct(mixed_list)
    assert res == {"list": ["int", "str", "str", "...4 total"]}


def test_list_in_list():  # update this one
    list_in_list = [1, [[1, 2], 3, "here"]]
    res = struct(list_in_list)
    assert res == {"list": ["int", {"list": [{"list": ["int", "int"]}, "int", "str"]}]}


def test_numpy_array():  # maybe include three?
    np_arr = np.array([1, 2, 3, 4, 5, 6, 7])
    res = struct(np_arr)
    assert res == {"ndarray": ["int32, shape=(7,)"]}


def test_tuple():
    test_tuple = (1, 2, 3, 4, 5, 6)
    res = struct(test_tuple)
    assert res == {"tuple": ["int", "int", "int", "...6 total"]}


def test_numpy_3d():
    np_array = np.zeros((5, 5, 5))
    res = struct(np_array)
    assert res == {"ndarray": ["float64, shape=(5, 5, 5)"]}


def test_empty_numpy():
    res = struct(np.array([]))
    assert res == {"ndarray": ["float64, shape=(0,)"]}


## TF Tensors


def test_tf_tensor():
    tensor = pd.Series(
        [
            [tf.constant(1, shape=(1))],
            [tf.constant(1, shape=(1))],
            [tf.constant(1, shape=(1))],
            [tf.constant(1, shape=(1))],
        ]
    )

    res = struct(tensor)

    assert (
        str(res)
        == "{'Series': [{'list': [{'EagerTensor': ['int32, shape=(1,)']}]}, {'list': [{'EagerTensor': ['int32,"
        " shape=(1,)']}]}, {'list': [{'EagerTensor': ['int32, shape=(1,)']}]}, '...4 total']}"
    )
