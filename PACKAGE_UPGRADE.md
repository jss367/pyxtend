## Adding New Package to PyPI

- Update version number in `src/pyxtend/_version.py`
- Upgrade build
- (Windows): `py -m pip install --upgrade build`
- (Mac/Linux): `python -m pip install --upgrade build`
- Build
- (Windows): `py -m build`
- (Mac/Linux): `python -m build`
- Upgrade twine
- (Windows): `py -m pip install --upgrade twine`
- (Mac/Linux): `python -m pip install --upgrade twine`
- Upload distribution packages
- (Windows): `py -m twine upload dist/*`
- (Mac/Linux): `python -m twine upload dist/*`
