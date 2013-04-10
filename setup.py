from __future__ import print_function
from distutils.core import setup, Extension

def check_dependences():
    dependences = [{'name': 'msmbuilder',
                    'url': 'https://github.com/simtk/msmbuilder',
                    },
                   {'name': 'numpy',
                    'url': 'http://docs.scipy.org/doc/numpy/user/install.html'
                    },
                   {'name': 'scipy',
                    'url': 'http://www.scipy.org/Installing_SciPy'}
                   ]
    for d in dependences:
        try:
            __import__(d['name'])
        except ImportError as e:
            print('#'*80)
            print('msmbuilder-KDML requires {name}'.format(**d))
            print('you can install it from here:')
            print('    {url}'.format(**d))
            print('#'*80)
            raise

def extensions():
    import numpy
    import os
    WRMSD = Extension('kdml/_WRMSD',
                      sources = ["kdml/WRMSD/theobald_rmsd.c",
                                 "kdml/WRMSD/theobald_rmsd_wrap.c"],
                      extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3","-fopenmp"],
                      extra_link_args=['-lgomp'],
                      include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')]
                      )
    return [WRMSD]

setupkw = dict(name='msmbuilder-KDML',
               version='1.0',
               author='Robert McGibbon',
               author_email='rmcgibbo@gmail.com',
               description='Kinetically discriminitory metric learning',
               long_description='',
               url='http://github.com/rmcgibbo/KDML',
               packages=['kdml'],
               scripts=['scripts/KDML.py'])


if __name__ == '__main__':
    check_dependences()
    ext_modules = extensions()
    setupkw['ext_modules'] = ext_modules
    setup(**setupkw)

      
