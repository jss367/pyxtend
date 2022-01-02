## Adding New Package to PyPI

* Update version in setup.cfg
* Upgrade build
 * (Windows): `py -m pip install --upgrade build`
 * (Mac/Linux): `python3 -m pip install --upgrade build`
* Build
 * (Windows): `py -m build`
* Upload distribution packages
 * (Windows): `py -m pip install --upgrade twine`
* Upload
 * (Windows): `py -m twine upload dist/*`
