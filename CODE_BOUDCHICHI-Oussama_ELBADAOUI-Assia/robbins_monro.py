# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama & EL BADAOUI Assia
# Robbins Monro Algorithm class

import numpy as np
import time
import numbers
from functools import wraps



class RobbinsMonro:
    """Robbins Monro Algorithm (RMA).
    :class:`RobbinsMonro`:

    Parameters
    ----------
    x_0       : numpy.ndarray or numbers.Number
                initial value of the RM procedure.
    gamma     : callable function
                step size sequence.
    H         : callable function
                the function appeatring in the expectation
                representation of the mean function.
    rng       : callable function
                random number generator
                p.s.: please make sure to make sure to set 
                the default number of generated numbers to 1.
    extra_args: *args keyword
                extra arguments to be plugged in H
           
    Examples
    --------
    >>> from robbins_monro import RobbinsMonro
    >>> import numpy as np
    >>> H = lambda theta, x, alpha: np.exp(x * np.sqrt(np.abs(theta))) - 2 - alpha
    >>> gamma = lambda n: 1. / (n + 1.)
            
    >>> rng = np.random.randn
    >>> x_0 = 0.
    >>> extra_args = lambda n: 0.
    >>> RM = RobbinsMonro(x_0, gamma, H, rng, extra_args)
    >>> RM.get_target(100_000)
    1.3882943611198906
    """
    def __init__(self, x_0, gamma, H, rng, extra_Hargs) -> None:
        self.init_value = x_0
        self.x = x_0
        self.H = H
        self.step = gamma
        self.rng = rng
        self.polyak = np.zeros_like(x_0)
        self.history_x = np.array([self.x])
        self.history_polyak = np.array([self.polyak])
        self.extra_args = extra_Hargs
        self.__check_input()
    
    def __timer(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'Eslaped time :  {total_time:.4f} seconds')
            return result
        return timeit_wrapper
    
    def __check_input(self) -> None:
        if not isinstance(self.x, (numbers.Number, np.ndarray)) : 
            raise ValueError('please enter x_0 of type numpy.ndarray')
    
             
        

    def __track(self) -> None:
        self.history_x = np.vstack((self.history_x, self.x))
        self.history_polyak = np.vstack((self.history_polyak, self.polyak))

    def __iteration(self, n: int) -> None:
            
            self.x = self.x - self.step(n) * self.H(self.x, self.rng(), self.extra_args(n))  
            self.polyak = self.polyak - (self.polyak - self.x) / n
            self.__track()
    
    def reset(self) -> None:
        self.x = self.init_value
        self.polyak = np.zeros_like(self.x)
        self.history_x = np.array([self.x])
        self.history_polyak = np.array([self.polyak])
    
    @__timer
    def get_target(self, n_iter: int) -> None:
         
        for n in range(1, n_iter):
             self.__iteration(n)
             #if n % 1000 == 0: print('iter: ', n, ';\t x: ', self.x)
    
    def print_vals(self) -> None:
         print('x: ', self.x, ';\t polyak: ', self.polyak)
        

