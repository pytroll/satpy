"""Test module for pp.coverage.
"""

import unittest
from pp.projector import Projector
import string
import numpy as np
from pyresample.utils import AreaNotFound

class TestProjector(unittest.TestCase):
    """Class for testing the Projector class.
    """

    proj = None

    def test_init(self):
        """Creation of coverage.
        """

        self.assertRaises(TypeError, Projector)

        self.proj = Projector("scan", "euro")

        self.assertEquals(self.proj.in_area.area_id, "scan")
        self.assertEquals(self.proj.out_area.area_id, "euro")

        
        self.assertRaises(TypeError, Projector, random_string(20))
        self.assertRaises(AreaNotFound,
                          Projector,
                          random_string(20),
                          random_string(20))
        
#     def test_project_array(self):
#         """Projection of an array.
#         """
#         self.proj = Projector("scan", "euro", precompute = True)

#         in_array = np.ones((512, 512))

#         res = self.proj.project_array(in_array)

#         comp = np.zeros((512, 512))
#         comp[78:334, 181:437] = 1

#         self.assert_(np.all(comp == res))
        
#         self.proj = Projector("scan", "euro", precompute = False)

#         in_array = np.ones((512, 512))

#         res = self.proj.project_array(in_array)

#         comp = np.zeros((512, 512))
#         comp[78:334, 181:437] = 1

#         self.assert_(np.all(comp == res))

#         in_array = np.ma.ones((512, 512))

#         res = self.proj.project_array(in_array)

#         self.assert_(isinstance(res, np.ma.core.MaskedArray))
        


def random_string(length, choices = string.letters):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    import random
    return "".join([random.choice(choices)
                    for i in range(length)])

if __name__ == '__main__':
    unittest.main()
