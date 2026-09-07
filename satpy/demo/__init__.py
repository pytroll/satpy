"""Demo data download helper functions.

Each ``get_*`` function below downloads files to a local directory and returns
a list of paths to those files. Some (not all) functions have multiple options
for how the data is downloaded (via the ``method`` keyword argument)
including:

- gcsfs:
    Download data from a public google cloud storage bucket using the
    ``gcsfs`` package.
- unidata_thredds:
    Access data using OpenDAP or similar method from Unidata's
    public THREDDS server
    (https://thredds.unidata.ucar.edu/thredds/catalog.html).
- uwaos_thredds:
    Access data using OpenDAP or similar method from the
    University of Wisconsin - Madison's AOS department's THREDDS server.
- http:
    A last resort download method when nothing else is available of a
    tarball or zip file from one or more servers available to the Satpy
    project.
- uw_arcdata:
    A network mount available on many servers at the Space Science
    and Engineering Center (SSEC) at the University of Wisconsin - Madison.
    This is method is mainly meant when tutorials are taught at the SSEC
    using a Jupyter Hub server.

To use these functions, do:

    >>> from satpy import Scene, demo
    >>> filenames = demo.get_us_midlatitude_cyclone_abi()
    >>> scn = Scene(reader='abi_l1b', filenames=filenames)

"""

from .abi_l1b import get_hurricane_florence_abi  # noqa: F401, I001
from .abi_l1b import get_us_midlatitude_cyclone_abi  # noqa: F401
from .ahi_hsd import download_typhoon_surigae_ahi  # noqa: F401
from .fci import download_fci_test_data  # noqa: F401
from .seviri_hrit import download_seviri_hrit_20180228_1500  # noqa: F401
from .viirs_sdr import get_viirs_sdr_20170128_1229  # noqa: F401
