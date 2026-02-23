import setuptools

try:
    from Cython.Build import cythonize
    import numpy as np
    _ext_modules = cythonize(
        ["intraday/_frame.pyx", "intraday/_processor.pyx"],
        compiler_directives={"language_level": "3"},
    )
    _include_dirs = [np.get_include()]
except Exception:
    _ext_modules = []
    _include_dirs = []

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    ext_modules=_ext_modules,
    include_dirs=_include_dirs,
    name='intraday-mod',
    version='0.0.236',
    author='Hung Nguyen',
    author_email='spideynolove@gmail.com',
    description='Exchange/Broker Simulation Environment for Intraday Trading Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='exchange broker trading gymnasium environment simulation episode',
    url='https://github.com/spideynolove/intraday',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Creative Commons',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'gymnasium',
        'numpy>=1.18.1',
        'arrow>=0.13.1',
        'feather-format>=0.4.1',
    ],
    extras_require={
        'pyglet': ['pyglet>=1.5.16'],
    },
)
