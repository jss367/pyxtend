import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import torch
from shapely.geometry import Polygon

# from src.pyxtend import struct # this fails on my mac
from pyxtend import struct  # this works on my mac


def test_string():
    result = struct("Hello, world!")
    assert result == "str"


def test_small_list():
    result = struct([1, 2, 3])
    assert result == {"list": ["int", "int", "int"]}


def test_small_list_examples():
    result = struct([1, 2, 3], examples=True)
    assert result == {"list": [1, 2, 3]}


def test_empty_list():
    result = struct([])
    assert result == {"list": []}


def test_small_set():
    small_set = {1, "this"}
    result = struct(small_set)
    assert result in [{"set": ["int", "str"]}, {"set": ["str", "int"]}]


def test_large_set():
    large_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    result = struct(large_set)
    assert result == {"set": ["int", "int", "int", "...10 total"]}


def test_large_set_examples():
    large_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    result = struct(large_set, examples=True)
    assert result == {"set": [1, 2, 3, "...10 total"]}


def test_empty_set():
    result = struct(set())
    assert result == {"set": []}


def test_empty_set_examples():
    result = struct(set(), examples=True)
    assert result == {"set": []}


def test_empty_range():
    result = struct(range(0))
    assert result == {"range": []}


def test_range():
    result = struct(range(10))
    assert result == {"range": ["int", "int", "int", "...10 total"]}


def test_long_list():
    long_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = struct(long_list)
    assert result == {"list": ["int", "int", "int", "...9 total"]}


def test_mixed_list():  # udpate this one
    mixed_list = [1, "this", "sam", [4, 3, "here"]]
    result = struct(mixed_list)
    assert result == {"list": ["int", "str", "str", "...4 total"]}


def test_mixed_list_examples():  # udpate this one
    mixed_list = [1, "this", "sam", [4, 3, "here"]]
    result = struct(mixed_list, examples=True)
    assert result == {"list": [1, "this", "sam", "...4 total"]}


def test_list_in_list():  # update this one
    list_in_list = [1, [[1, 2], 3, "here"]]
    result = struct(list_in_list)
    assert result == {"list": ["int", {"list": [{"list": ["int", "int"]}, "int", "str"]}]}


def test_numpy_array():  # maybe include three?
    np_arr = np.array([1, 2, 3, 4, 5, 6, 7])
    result = struct(np_arr)
    # Reponse will depend on default interger type of the platform
    expected_dtype = "int32" if np.dtype("int").itemsize == 4 else "int64"
    assert result == {"ndarray": [f"{expected_dtype}, shape=(7,)"]}


def test_large_numpy_array():
    np_arr = np.zeros((1000, 2, 256, 256, 3))  # reduced from (10000, 2, 256, 256, 3) to pass build
    result = struct(np_arr)
    # Reponse will depend on default interger type of the platform
    expected_dtype = "float32" if np.dtype("float").itemsize == 4 else "float64"
    assert result == {"ndarray": [f"{expected_dtype}, shape=(10000, 2, 256, 256, 3)"]}


def test_numpy_array_examples():  # maybe include three?
    np_arr = np.array([1, 2, 3, 4, 5, 6, 7])
    result = struct(np_arr, examples=True)
    # Reponse will depend on default interger type of the platform
    expected_dtype = "int32" if np.dtype("int").itemsize == 4 else "int64"
    assert result == {"ndarray": [f"{expected_dtype}, shape=(7,)"]}


def test_dict():
    result = struct(({"a": 1, "b": 2}))
    assert result == {"dict": ["str", "str"]}


def test_large_dict():
    result = struct(({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}))
    assert result == {"dict": ["str", "str", "str", "...5 total"]}


def test_tuple():
    medium_tuple = (1, 2, 3, 4, 5, 6)
    result = struct(medium_tuple)
    assert result == {"tuple": ["int", "int", "int", "...6 total"]}


def test_numpy_3d():
    np_array = np.zeros((5, 5, 5))
    result = struct(np_array)
    assert result == {"ndarray": ["float64, shape=(5, 5, 5)"]}


def test_empty_numpy():
    result = struct(np.array([]))
    assert result == {"ndarray": ["float64, shape=(0,)"]}


# PyTorch Tensors


def test_torch_tensor():
    torch_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    result = struct(torch_tensor)
    assert result == {"Tensor": ["torch.float32, shape=(2, 3)"]}


def test_larger_torch_tensor():
    torch_tensor = torch.Tensor(128, 3, 256, 256)
    result = struct(torch_tensor)
    assert result == {"Tensor": ["torch.float32, shape=(128, 3, 256, 256)"]}


# TF Tensors


def test_tf_tensor():
    tf_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
    result = struct(tf_tensor)
    assert result == {"EagerTensor": ["<dtype: 'float32'>, shape=(2, 3)"]}


def test_tf_tensors():
    tensor = pd.Series(
        [
            [tf.constant(1, shape=1)],
            [tf.constant(1, shape=1)],
            [tf.constant(1, shape=1)],
            [tf.constant(1, shape=1)],
        ]
    )

    result = struct(tensor)

    assert (
        str(result)
        == "{'Series': [{'list': [{'EagerTensor': [\"<dtype: 'int32'>, shape=(1,)\"]}]}, {'list': [{'EagerTensor':"
        " [\"<dtype: 'int32'>, shape=(1,)\"]}]}, {'list': [{'EagerTensor': [\"<dtype: 'int32'>, shape=(1,)\"]}]},"
        " '...4 total']}"
    )


def test_shapely_polygon():
    polygon = Polygon([(0, 0), (1, 1), (1, 0)])
    result = struct(polygon)
    assert result == {"Polygon": ["float64, shape=(4, 2)"]}  # starting point is added as end point


# Make some custom classes for testing


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Shape:
    def __init__(self, name, points):
        self.name = name
        self.points = points


class MyClass:
    def __init__(self, a, b, some_list):
        self.attribute1 = a
        self.attribute2 = b
        self.attribute3 = some_list
        self.my_method = lambda x: x * 2


def test_simple_object():
    point = Point(10, 20)
    result = struct(point)
    expected = {"Point": {"x": "int", "y": "int"}}
    assert result == expected


def test_nested_objects():
    shape = Shape("rectangle", [Point(0, 0), Point(10, 0), Point(10, 5), Point(0, 5)])
    result = struct(shape)
    expected = {
        "Shape": {
            "name": "str",
            "points": {
                "list": [
                    {"Point": {"x": "int", "y": "int"}},
                    {"Point": {"x": "int", "y": "int"}},
                    {"Point": {"x": "int", "y": "int"}},
                    "...4 total",
                ]
            },
        }
    }
    assert result == expected  # You'll need to define the expected output


def test_mixed_types():
    my_obj = MyClass(5, "hello", [1, 2.5, "world"])
    result = struct(my_obj)
    expected = {
        "MyClass": {
            "attribute1": "int",
            "attribute2": "str",
            "attribute3": {"list": ["int", "float", "str"]},
            "my_method": {"function": {}},
        }
    }
    assert result == expected


# Test the groupby_summary function


# Sample DataFrame
data = {
    "group": ["A", "A", "B", "B", "B"],
    "value1": [10, 20, 5, 15, 8],
    "value2": ["hello", "world", "foo", "bar", "baz"],
}
df = pd.DataFrame(data)
groupby_obj = df.groupby("group")


def test_groupby_summary():
    result = struct(groupby_obj)

    # Expected Results (you'll need to verify and adjust these based on your data)
    expected = {
        "total_items": 5,
        "num_groups": 2,
        "average_items_per_group": 2.5,
        "group_size_stats": {
            "count": 2.0,
            "mean": 2.5,
            "std": pytest.approx(0.707106, abs=1e-6),
            "min": 2.0,
            "25%": 2.25,
            "50%": 2.5,
            "75%": 2.75,
            "max": 3.0,
        },
        "group_examples": {
            "A": pd.DataFrame({"group": ["A", "A"], "value1": [10, 20], "value2": ["hello", "world"]}),
            "B": pd.DataFrame({"group": ["B", "B"], "value1": [5, 15], "value2": ["foo", "bar"]}),
        },
        "mean_values": pd.Series({"value1": pytest.approx(12.1666, abs=1e-1)}),
        "median_values": pd.Series({"value1": 11.5}),
    }

    # Ensure all expected keys are present
    assert set(result.keys()) == set(expected.keys())

    # Compare numeric values
    for key in ["total_items", "num_groups", "average_items_per_group"]:
        assert result[key] == expected[key]

    # Compare 'group_size_stats'
    assert result["group_size_stats"] == expected["group_size_stats"]

    # Compare 'group_examples' (check DataFrames for equality)
    for group_name in result["group_examples"]:
        assert (
            result["group_examples"][group_name]
            .reset_index(drop=True)
            .equals(expected["group_examples"][group_name].reset_index(drop=True))
        )

    # Compare 'mean_values' & 'median_values' (check Series for equality)
    assert (
        result["mean_values"].values[0] == expected["mean_values"].values[0]
    )  # Have to do it this way because of pytest approx
    assert result["median_values"].equals(expected["median_values"])
