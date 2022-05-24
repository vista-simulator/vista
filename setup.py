#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [ ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Alexander Amini",
    author_email='amini@mit.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Data-driven simulation for training and evaluating full-scale autonomous vehicles",
    long_description=readme,
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='vista',
    name='vista',
    packages=find_packages(include=['vista', 'vista.*']),
    setup_requires=setup_requirements,
    url='https://github.com/vista-simulator/vista',
    download_url = 'https://github.com/vista-simulator/vista/archive/refs/tags/2.0.6.tar.gz',
    version='2.0.6',
    zip_safe=False,
)
