import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='intraday',    # intraday-mod
    version='0.0.236',  # depend on you
    author='Pavel B. Chernov',  # Hung Nguyen
    author_email='pavel.b.chernov@gmail.com',   # 'spideynolove@gmail.com' not 'pavel.b.chernov@gmail.com
    description='Exchange/Broker Simulation Environment for Intraday Trading Models',   # depend on you
    long_description=long_description,  # ???? depend on you
    long_description_content_type='text/markdown',  # depend on you
    keywords='exchange broker trading gym environment simulation episode',  # depend on you
    url='https://github.com/diovisgood/intraday',   # 'spideynolove' not 'diovisgood'
    packages=setuptools.find_packages(),        # use new 'uv astral' not  'setuptools
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Creative Commons',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',            # >=3.10
    install_requires=[
        'gym>=0.17.2',  # -> 'gymnasium' not 'gym'
        'numpy>=1.18.1',    # depend on you
        'arrow>=0.13.1',    # depend on you
        'feather-format>=0.4.1',    # depend on you
        # T.B.D # maybe we need more libraries, depend on me and you
],
    extras_require={
        'pyglet': ['pyglet>=1.5.16'],
    },
)
