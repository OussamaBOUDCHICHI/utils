# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama & EL BADAOUI Assia
# A simple gsl wrapper, to use Sobol sequences generator of C++ library : GSL, in python.

import cython
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Declare the few types and functions we need
cdef extern from "gsl/gsl_qrng.h":
    ctypedef struct gsl_qrng
    ctypedef struct gsl_qrng_type
    gsl_qrng_type* gsl_qrng_sobol
    
    gsl_qrng* gsl_qrng_alloc(gsl_qrng_type* T, unsigned int d)
    void gsl_qrng_free(gsl_qrng* q)
    
    int gsl_qrng_get(const gsl_qrng * q, double x[])

# This is the wrapper class
cdef class Sobol:
    cdef gsl_qrng* _q
    cdef unsigned _D
    cdef double *_v
    def __cinit__(self, D=1):
        """Create a `D` dimensional Sobol sequence."""
        self._D = D
        
        # gsl_qrng_get() returns the next
        # value in one of its arguments
        self._v = <double *>PyMem_Malloc(D * sizeof(double))
        if not self._v:
            raise MemoryError()
        
        # Actually allocate the QRNG generator
        self._q = gsl_qrng_alloc(gsl_qrng_sobol, D)
        if self._q is NULL:
            raise MemoryError()
        
    def get(self, int N = 1):
        """The next `N` points in the sequence."""
        points = []
        for n in xrange(N):
            points.append(self.__next__())
            
        return np.array(points)
    
    def get_1dgaussian(self, int N = 1):
        sobol_seq = self.get(N)
        out = np.zeros_like(sobol_seq)
        
        for k in range(out.shape[0]):
            out[k, 0] = np.sqrt(-2 * np.log(sobol_seq[k, 0])) * np.cos(2 * np.pi * sobol_seq[k, 1])
            out[k, 1] = np.sqrt(-2 * np.log(sobol_seq[k, 0])) * np.sin(2 * np.pi * sobol_seq[k, 1])
        
        Z = np.sum(out, axis = 1)
        if N == 1:
            return Z[0]
        else:
            return Z
        
    def __next__(self):
        """Iterate over the sequence."""
        gsl_qrng_get(self._q, self._v)
        return [self._v[i] for i in xrange(self._D)]
    
    def __iter__(self):
        return self
    
    # Make sure we free all the memory we allocated
    def __dealloc__(self):
        if self._q is not NULL:
            gsl_qrng_free(self._q)
            
        PyMem_Free(self._v)
