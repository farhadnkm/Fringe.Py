cd dist
sdelete -s *
cd..
cd build
sdelete -s *
cd..
python setup.py bdist_wheel sdist
pip install -e .
twine upload --skip-existing dist/*
