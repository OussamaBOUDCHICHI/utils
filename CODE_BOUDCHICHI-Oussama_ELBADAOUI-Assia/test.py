# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama & EL BADAOUI Assia
# Unitest of class : robbins_monro.RobbinsMonro
# function x -> h(x) := E(e^{sqrt(|x|) * X} - 2.), where X ~ N(0, 1).
# explicit formula of the zero of h: 2.log(2.)

import unittest
from robbins_monro import RobbinsMonro
import numpy as np


class TestRM(unittest.TestCase):
    def test_result(self):

        H = lambda theta, x, alpha: np.exp(x * np.sqrt(np.abs(theta))) - 2 - alpha
        gamma = lambda n: 1. / (n + 1.)
            
        rng = np.random.randn
        x_0 = 0.
        extra_args = lambda n: 0.
        RM = RobbinsMonro(x_0, gamma, H, rng, extra_args)
        RM.get_target(100_000)
        
        self.assertAlmostEqual(RM.x, 2. * np.log(2.), delta = 2e-2) and self.assertAlmostEqual(RM.polyak, 2. * np.log(2.), delta = 2e-2)


if __name__ == '__main__':
    unittest.main()
