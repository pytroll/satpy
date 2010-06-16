"""Running all unit tests.
"""

import sys

try:
    import nose
except ImportError:
    print ('nose is required to run the test suite')
    sys.exit(1)

nose.main()

