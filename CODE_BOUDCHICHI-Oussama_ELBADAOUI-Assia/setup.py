from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Make sure to change the following paths with the path to gsl in your device.
gsl_include_dir = '/opt/homebrew/Cellar/gsl/2.7.1/include'
gsl_library_dir = '/opt/homebrew/Cellar/gsl/2.7.1/lib'


ext = Extension('_sobol_cy', ['sobol_gsl.pyx'], 
                include_dirs=[gsl_include_dir], 
                library_dirs=[gsl_library_dir], 
                libraries=["gsl"])

setup(
  name='Sobol Sequence Generator',
  ext_modules=[ext],
  cmdclass={'build_ext': build_ext},
)