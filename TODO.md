## TODO
* Needs better performance on cases like np.zeros((10000, 2, 256, 256, 3))
  * Should it only do the extra thing for sets?

If it's unsupported, should it be:
return "unsupported"  # type(obj).__name__ or {type(obj).__name__: 'unsupported'}
