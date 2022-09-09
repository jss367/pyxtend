import numpy as np
from src.pyxtend import struct

small_set = {1, "this"}
res = struct(small_set)
# the order here is not determined, so it could go either way
# assert str(res) == "{<class 'set'>: [<class 'int'>, <class 'str'>]}"
# "{<class 'set'>: [<class 'str'>, <class 'int'>]}"

large_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
res = struct(large_set)
assert str(res) == "{<class 'set'>: [<class 'int'>, <class 'int'>, <class 'int'>, '...10 total']}"
"{<class 'set'>: [<class 'int'>, <class 'int'>, <class 'int'>]}"

res = struct(range(10))
assert str(res) == "{<class 'range'>: [<class 'int'>, <class 'int'>, <class 'int'>, '...10 total']}"

res = struct(range(0))
assert str(res) == "{<class 'range'>: []}"

long_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
res = struct(long_list)
assert str(res) == "{<class 'list'>: [<class 'int'>, <class 'int'>, <class 'int'>, '...9 total']}"


mixed_list = [1, "this", "sam", [4, 3, "here"]]
res = struct(mixed_list)
assert str(res) == "{<class 'list'>: [<class 'int'>, <class 'str'>, <class 'str'>, '...4 total']}"


recursive_list = [1, [[1, 2], 3, "here"]]
res = struct(recursive_list)
assert (
    str(res)
    == "{<class 'list'>: [<class 'int'>, {<class 'list'>: [{<class 'list'>: [<class 'int'>, <class 'int'>]}, <class 'int'>, <class 'str'>]}]}"
)


recursive_list = np.array([1, 2, 3, 4, 5, 6, 7])
res = struct(recursive_list)
assert (
    str(res)
    == "{<class 'numpy.ndarray'>: [<class 'numpy.int32'>, <class 'numpy.int32'>, <class 'numpy.int32'>, '...7 total']}"
)

# np_array = np.zeros((5, 5, 5))
# res = struct(np_array)
# assert (
#     str(res)
#     == "{<class 'numpy.ndarray'>: [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, '...5 total']}"
# )
# "{<class 'numpy.ndarray'>: [{<class 'numpy.ndarray'>: [{<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, {<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, {<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, '...5 total']}, {<class 'numpy.ndarray'>: [{<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, {<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, {<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, '...5 total']}, {<class 'numpy.ndarray'>: [{<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, {<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, {<class 'numpy.ndarray'>: [<class 'numpy.float64'>, <class 'numpy.float64'>, <class 'numpy.float64'>, '...5 total']}, '...5 total']}, '...5 total']}"

# huge_array = np.zeros((10000, 5, 256, 256, 3))
# res = struct(huge_array)
# assert (
#     str(res)
#     == "{<class 'numpy.ndarray'>: [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, '...5 total']}"
# )


## TF Tensors

tensor = pd.Series([[tf.constant(1, shape=(1))],[tf.constant(1, shape=(1))],[tf.constant(1, shape=(1))], [tf.constant(1, shape=(1))]])

res = struct(tensor)

assert res == {pd.Series: [{list: [tf.TensorShape([1])]}, {list: [tf.TensorShape([1])]}, {list: [tf.TensorShape([1])]}, '...4 total']}


