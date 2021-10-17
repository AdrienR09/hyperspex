from setuptools import setup

setup(
    name='hyperspex',
    version='1.0.1',
    description='A Python package to help 3D hyperspectral data exploration through a Graphical User Interface.',
    url='https://github.com/AdrienR09/hyperspex/',
    author='Adrien Rousseau',
    author_email='adrien.rousseau@umontpellier.fr',
    license='BSD 2-clause',
    packages=[],
    install_requires=['mpi4py>=2.0',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)