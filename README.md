# pyxtend

Some functions to make Python more productive.

## TODO
* Add numpy to requirements
* Add rest of functions
* Needs better performance on cases like np.zeros((10000, 2, 256, 256, 3))
  * Should it only do the extra thing for sets?
* Should it provide an optional example in some cases?
* Add `example_depth` argument, where default is 0 but value of 1 means example is provided for top layer, value of 2 for top two, etc.
  * Can also have an `auto` input, where it's done smartly (i.e. if the example is going to be huge, don't display it)
