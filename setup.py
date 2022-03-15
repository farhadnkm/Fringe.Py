#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as req_file:
    requirements = req_file.read()

setup(
    name='fringe',
    version='1.0.2',
    description="Python implementation of computational optical propagation and digital holography.",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    url='https://github.com/farhadnkm/fringe',
    author="Farhad Niknam",
    author_email='farhad.niknam.em@gmail.com',
    python_requires='>=3.6, <4',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Image Processing'
    ],
    keywords='diffraction, angular spectrum, holography, holograms, phase recovery,',
    #package_dir={'': 'src'},
    packages=find_packages(),
    license="MIT license",
    install_requires=requirements,
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest', 'jupyter'],
    test_suite='tests',
    zip_safe=False,
)
