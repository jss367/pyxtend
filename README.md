# pyxtend

Functions to be more productive in Python.

## struct

`struct` is for examining objects to understand their contents.

print(struct("Hello, world!"))
str

print(struct([1, 2, 3]))
{'list': ['int', 'int', 'int']}

print(struct({"a": 1, "b": 2}))
{'dict': ['str', 'str']}

print(struct(set([1, 2, 3])))
{'set': ['int', 'int', 'int']}

print(struct(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)))
{'ndarray': ['float32, shape=(2, 3)']}

print(struct(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)))
{'Tensor': ['torch.float32, shape=(2, 3)']}

print(struct(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)))
{'EagerTensor': ['float32, shape=(2, 3)']}


## vprint
