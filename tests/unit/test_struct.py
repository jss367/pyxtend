import numpy as np
from src.pyxtend.pyxtend import struct

small_set = {1, "this"}
res = struct(small_set)
assert str(res) == "{<class 'set'>: [<class 'int'>, <class 'str'>]}"

large_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
res = struct(large_set)
assert str(res) == "{<class 'set'>: [<class 'int'>, <class 'int'>, <class 'int'>, '...10 total']}"

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
assert str(res) == "{<class 'list'>: [<class 'int'>, <class 'list'>]}"

recursive_list = np.array([1, 2, 3, 4, 5, 6, 7])
res = struct(recursive_list)
assert (
    str(res)
    == "{<class 'numpy.ndarray'>: [<class 'numpy.int32'>, <class 'numpy.int32'>, <class 'numpy.int32'>, '...7 total']}"
)

recursive_list = np.zeros((5, 5, 5))
res = struct(recursive_list)
assert (
    str(res)
    == "{<class 'numpy.ndarray'>: [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, '...5 total']}"
)
