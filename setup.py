from setuptools import setup, find_packages
from os import path

with open(path.join('README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "sc2-env",
    packages = find_packages(),
    version = "0.11.1.1",
    description = "A StarCraft II bot gym env library over python-sc2",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    author = "Damien GILLES",
    author_email = "damien.gilles.pro@gmail.com",
    url = "https://github.com/gillesdami/python-sc2",
    keywords = ["StarCraft", "StarCraft 2", "StarCraft II", "AI", "Bot"],
    python_requires='>=3.6',
    install_requires=['portpicker', 'async-timeout>=3.0,<4.0', 's2clientprotocol', 'aiohttp', 'pyglet', 'gym'],
    classifiers = [
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Games/Entertainment",
        "Topic :: Games/Entertainment :: Real Time Strategy",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ]
)
