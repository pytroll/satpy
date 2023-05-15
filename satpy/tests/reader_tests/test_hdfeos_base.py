#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for the HDF-EOS base functionality."""

import unittest

nrt_mda = '''GROUP                  = INVENTORYMETADATA
  GROUPTYPE            = MASTERGROUP

  GROUP                  = ECSDATAGRANULE

    OBJECT                 = REPROCESSINGPLANNED
      NUM_VAL              = 1
      VALUE                = "further update is anticipated"
    END_OBJECT             = REPROCESSINGPLANNED

    OBJECT                 = REPROCESSINGACTUAL
      NUM_VAL              = 1
      VALUE                = "Near Real Time"
    END_OBJECT             = REPROCESSINGACTUAL

    OBJECT                 = LOCALGRANULEID
      NUM_VAL              = 1
      VALUE                = "MYD03.A2019051.1225.061.2019051131153.NRT.hdf"
    END_OBJECT             = LOCALGRANULEID

    OBJECT                 = PRODUCTIONDATETIME
      NUM_VAL              = 1
      VALUE                = "2019-02-20T13:11:53.000Z"
    END_OBJECT             = PRODUCTIONDATETIME

    OBJECT                 = DAYNIGHTFLAG
      NUM_VAL              = 1
      VALUE                = "Day"
    END_OBJECT             = DAYNIGHTFLAG

    OBJECT                 = LOCALVERSIONID
      NUM_VAL              = 1
      VALUE                = "6.0.4"
    END_OBJECT             = LOCALVERSIONID

  END_GROUP              = ECSDATAGRANULE

  GROUP                  = MEASUREDPARAMETER

    OBJECT                 = MEASUREDPARAMETERCONTAINER
      CLASS                = "1"

      OBJECT                 = PARAMETERNAME
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = "Geolocation"
      END_OBJECT             = PARAMETERNAME

      GROUP                  = QAFLAGS
        CLASS                = "1"

        OBJECT                 = AUTOMATICQUALITYFLAG
          NUM_VAL              = 1
          CLASS                = "1"
          VALUE                = "Passed"
        END_OBJECT             = AUTOMATICQUALITYFLAG

        OBJECT                 = AUTOMATICQUALITYFLAGEXPLANATION
          NUM_VAL              = 1
          CLASS                = "1"
          VALUE                = "Set to 'Failed' if processing error occurred, set to 'Passed' otherwise"
        END_OBJECT             = AUTOMATICQUALITYFLAGEXPLANATION

        OBJECT                 = SCIENCEQUALITYFLAG
          NUM_VAL              = 1
          VALUE                = "Not Investigated"
          CLASS                = "1"
        END_OBJECT             = SCIENCEQUALITYFLAG

      END_GROUP              = QAFLAGS

      GROUP                  = QASTATS
        CLASS                = "1"

        OBJECT                 = QAPERCENTMISSINGDATA
          NUM_VAL              = 1
          CLASS                = "1"
          VALUE                = 0
        END_OBJECT             = QAPERCENTMISSINGDATA

        OBJECT                 = QAPERCENTOUTOFBOUNDSDATA
          NUM_VAL              = 1
          CLASS                = "1"
          VALUE                = 0
        END_OBJECT             = QAPERCENTOUTOFBOUNDSDATA

      END_GROUP              = QASTATS

    END_OBJECT             = MEASUREDPARAMETERCONTAINER

  END_GROUP              = MEASUREDPARAMETER

  GROUP                  = ORBITCALCULATEDSPATIALDOMAIN

    OBJECT                 = ORBITCALCULATEDSPATIALDOMAINCONTAINER
      CLASS                = "1"

      OBJECT                 = ORBITNUMBER
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = 89393
      END_OBJECT             = ORBITNUMBER

      OBJECT                 = EQUATORCROSSINGLONGITUDE
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = -151.260740805733
      END_OBJECT             = EQUATORCROSSINGLONGITUDE

      OBJECT                 = EQUATORCROSSINGTIME
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = "12:49:52.965727"
      END_OBJECT             = EQUATORCROSSINGTIME

      OBJECT                 = EQUATORCROSSINGDATE
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = "2019-02-20"
      END_OBJECT             = EQUATORCROSSINGDATE

    END_OBJECT             = ORBITCALCULATEDSPATIALDOMAINCONTAINER

  END_GROUP              = ORBITCALCULATEDSPATIALDOMAIN

  GROUP                  = COLLECTIONDESCRIPTIONCLASS

    OBJECT                 = SHORTNAME
      NUM_VAL              = 1
      VALUE                = "MYD03"
    END_OBJECT             = SHORTNAME

    OBJECT                 = VERSIONID
      NUM_VAL              = 1
      VALUE                = 61
    END_OBJECT             = VERSIONID

  END_GROUP              = COLLECTIONDESCRIPTIONCLASS

  GROUP                  = INPUTGRANULE

    OBJECT                 = INPUTPOINTER
      NUM_VAL              = 8
      VALUE                = ("MYD01.61.2019-051T12:25:00.000000Z.NA.29878844.500100_1.hdf", "MYD03LUT.coeff_V6.1.4", "PM1EPHND_NRT.A2019051.1220.061.2019051125628", "PM1EPHND_NRT.A2019051.1225.061.2019051125628", "PM1EPHND_NRT.A2019051.1230.061.2019051125628", "
          PM1ATTNR_NRT.A2019051.1220.061.2019051125628", "PM1ATTNR_NRT.A2019051.1225.061.2019051125628", "PM1ATTNR_NRT.A2019051.1230.061.2019051125628")
    END_OBJECT             = INPUTPOINTER

  END_GROUP              = INPUTGRANULE

  GROUP                  = SPATIALDOMAINCONTAINER

    GROUP                  = HORIZONTALSPATIALDOMAINCONTAINER

      GROUP                  = GPOLYGON

        OBJECT                 = GPOLYGONCONTAINER
          CLASS                = "1"

          GROUP                  = GRING
            CLASS                = "1"

            OBJECT                 = EXCLUSIONGRINGFLAG
              NUM_VAL              = 1
              CLASS                = "1"
              VALUE                = "N"
            END_OBJECT             = EXCLUSIONGRINGFLAG

          END_GROUP              = GRING

          GROUP                  = GRINGPOINT
            CLASS                = "1"

            OBJECT                 = GRINGPOINTLONGITUDE
              NUM_VAL              = 4
              CLASS                = "1"
              VALUE                = (25.3839329817764, 1.80418778807854, -6.50842421663422, 23.0260060198343)
            END_OBJECT             = GRINGPOINTLONGITUDE

            OBJECT                 = GRINGPOINTLATITUDE
              NUM_VAL              = 4
              CLASS                = "1"
              VALUE                = (29.5170117594673, 26.1480434828114, 43.2445462598877, 47.7959787025408)
            END_OBJECT             = GRINGPOINTLATITUDE

            OBJECT                 = GRINGPOINTSEQUENCENO
              NUM_VAL              = 4
              CLASS                = "1"
              VALUE                = (1, 2, 3, 4)
            END_OBJECT             = GRINGPOINTSEQUENCENO

          END_GROUP              = GRINGPOINT

        END_OBJECT             = GPOLYGONCONTAINER

      END_GROUP              = GPOLYGON

    END_GROUP              = HORIZONTALSPATIALDOMAINCONTAINER

  END_GROUP              = SPATIALDOMAINCONTAINER

  GROUP                  = RANGEDATETIME

    OBJECT                 = RANGEBEGINNINGTIME
      NUM_VAL              = 1
      VALUE                = "12:25:00.000000"
    END_OBJECT             = RANGEBEGINNINGTIME

    OBJECT                 = RANGEENDINGTIME
      NUM_VAL              = 1
      VALUE                = "12:30:00.000000"
    END_OBJECT             = RANGEENDINGTIME

    OBJECT                 = RANGEBEGINNINGDATE
      NUM_VAL              = 1
      VALUE                = "2019-02-20"
    END_OBJECT             = RANGEBEGINNINGDATE

    OBJECT                 = RANGEENDINGDATE
      NUM_VAL              = 1
      VALUE                = "2019-02-20"
    END_OBJECT             = RANGEENDINGDATE

  END_GROUP              = RANGEDATETIME

  GROUP                  = ASSOCIATEDPLATFORMINSTRUMENTSENSOR

    OBJECT                 = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER
      CLASS                = "1"

      OBJECT                 = ASSOCIATEDSENSORSHORTNAME
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = "MODIS"
      END_OBJECT             = ASSOCIATEDSENSORSHORTNAME

      OBJECT                 = ASSOCIATEDPLATFORMSHORTNAME
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = "Aqua"
      END_OBJECT             = ASSOCIATEDPLATFORMSHORTNAME

      OBJECT                 = ASSOCIATEDINSTRUMENTSHORTNAME
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = "MODIS"
      END_OBJECT             = ASSOCIATEDINSTRUMENTSHORTNAME

    END_OBJECT             = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER

  END_GROUP              = ASSOCIATEDPLATFORMINSTRUMENTSENSOR

  GROUP                  = PGEVERSIONCLASS

    OBJECT                 = PGEVERSION
      NUM_VAL              = 1
      VALUE                = "6.1.4"
    END_OBJECT             = PGEVERSION

  END_GROUP              = PGEVERSIONCLASS

  GROUP                  = ADDITIONALATTRIBUTES

    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER
      CLASS                = "1"

      OBJECT                 = ADDITIONALATTRIBUTENAME
        CLASS                = "1"
        NUM_VAL              = 1
        VALUE                = "GRANULENUMBER"
      END_OBJECT             = ADDITIONALATTRIBUTENAME

      GROUP                  = INFORMATIONCONTENT
        CLASS                = "1"

        OBJECT                 = PARAMETERVALUE
          NUM_VAL              = 1
          CLASS                = "1"
          VALUE                = "151"
        END_OBJECT             = PARAMETERVALUE

      END_GROUP              = INFORMATIONCONTENT

    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER

    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER
      CLASS                = "2"

      OBJECT                 = ADDITIONALATTRIBUTENAME
        CLASS                = "2"
        NUM_VAL              = 1
        VALUE                = "SCI_STATE"
      END_OBJECT             = ADDITIONALATTRIBUTENAME

      GROUP                  = INFORMATIONCONTENT
        CLASS                = "2"

        OBJECT                 = PARAMETERVALUE
          NUM_VAL              = 1
          CLASS                = "2"
          VALUE                = "1"
        END_OBJECT             = PARAMETERVALUE

      END_GROUP              = INFORMATIONCONTENT

    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER

    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER
      CLASS                = "3"

      OBJECT                 = ADDITIONALATTRIBUTENAME
        CLASS                = "3"
        NUM_VAL              = 1
        VALUE                = "SCI_ABNORM"
      END_OBJECT             = ADDITIONALATTRIBUTENAME

      GROUP                  = INFORMATIONCONTENT
        CLASS                = "3"

        OBJECT                 = PARAMETERVALUE
          NUM_VAL              = 1
          CLASS                = "3"
          VALUE                = "1"
        END_OBJECT             = PARAMETERVALUE

      END_GROUP              = INFORMATIONCONTENT

    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER

    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER
      CLASS                = "5"

      OBJECT                 = ADDITIONALATTRIBUTENAME
        CLASS                = "5"
        NUM_VAL              = 1
        VALUE                = "PROCESSVERSION"
      END_OBJECT             = ADDITIONALATTRIBUTENAME

      GROUP                  = INFORMATIONCONTENT
        CLASS                = "5"

        OBJECT                 = PARAMETERVALUE
          NUM_VAL              = 1
          CLASS                = "5"
          VALUE                = "6.1.0"
        END_OBJECT             = PARAMETERVALUE

      END_GROUP              = INFORMATIONCONTENT

    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER

    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER
      CLASS                = "4"

      OBJECT                 = ADDITIONALATTRIBUTENAME
        CLASS                = "4"
        NUM_VAL              = 1
        VALUE                = "GEO_EST_RMS_ERROR"
      END_OBJECT             = ADDITIONALATTRIBUTENAME

      GROUP                  = INFORMATIONCONTENT
        CLASS                = "4"

        OBJECT                 = PARAMETERVALUE
          NUM_VAL              = 1
          CLASS                = "4"
          VALUE                = "75      "
        END_OBJECT             = PARAMETERVALUE

      END_GROUP              = INFORMATIONCONTENT

    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER

    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER
      CLASS                = "6"

      OBJECT                 = ADDITIONALATTRIBUTENAME
        CLASS                = "6"
        NUM_VAL              = 1
        VALUE                = "identifier_product_doi"
      END_OBJECT             = ADDITIONALATTRIBUTENAME

      GROUP                  = INFORMATIONCONTENT
        CLASS                = "6"

        OBJECT                 = PARAMETERVALUE
          NUM_VAL              = 1
          CLASS                = "6"
          VALUE                = "10.5067/MODIS/MYD03.NRT.061"
        END_OBJECT             = PARAMETERVALUE

      END_GROUP              = INFORMATIONCONTENT

    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER

    OBJECT                 = ADDITIONALATTRIBUTESCONTAINER
      CLASS                = "7"

      OBJECT                 = ADDITIONALATTRIBUTENAME
        CLASS                = "7"
        NUM_VAL              = 1
        VALUE                = "identifier_product_doi_authority"
      END_OBJECT             = ADDITIONALATTRIBUTENAME

      GROUP                  = INFORMATIONCONTENT
        CLASS                = "7"

        OBJECT                 = PARAMETERVALUE
          NUM_VAL              = 1
          CLASS                = "7"
          VALUE                = "http://dx.doi.org"
        END_OBJECT             = PARAMETERVALUE

      END_GROUP              = INFORMATIONCONTENT

    END_OBJECT             = ADDITIONALATTRIBUTESCONTAINER

  END_GROUP              = ADDITIONALATTRIBUTES

END_GROUP              = INVENTORYMETADATA

END'''  # noqa: E501

nrt_mda_dict = {
    'INVENTORYMETADATA': {
        'ADDITIONALATTRIBUTES': {
            'ADDITIONALATTRIBUTESCONTAINER': {
                'ADDITIONALATTRIBUTENAME': {
                    'VALUE': 'identifier_product_doi_authority'
                },
                'INFORMATIONCONTENT': {
                    'PARAMETERVALUE': {
                        'VALUE': 'http://dx.doi.org'
                    }
                }
            }
        },
        'ASSOCIATEDPLATFORMINSTRUMENTSENSOR': {
            'ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER': {
                'ASSOCIATEDINSTRUMENTSHORTNAME': {
                    'VALUE': 'MODIS'
                },
                'ASSOCIATEDPLATFORMSHORTNAME': {
                    'VALUE': 'Aqua'
                },
                'ASSOCIATEDSENSORSHORTNAME': {
                    'VALUE': 'MODIS'
                }
            }
        },
        'COLLECTIONDESCRIPTIONCLASS': {
            'SHORTNAME': {
                'VALUE': 'MYD03'
            },
            'VERSIONID': {
                'VALUE': 61
            }
        },
        'ECSDATAGRANULE': {
            'DAYNIGHTFLAG': {
                'VALUE': 'Day'
            },
            'LOCALGRANULEID': {
                'VALUE': 'MYD03.A2019051.1225.061.2019051131153.NRT.hdf'
            },
            'LOCALVERSIONID': {
                'VALUE': '6.0.4'
            },
            'PRODUCTIONDATETIME': {
                'VALUE': '2019-02-20T13:11:53.000Z'
            },
            'REPROCESSINGACTUAL': {
                'VALUE': 'Near '
                'Real '
                'Time'
            },
            'REPROCESSINGPLANNED': {
                'VALUE': 'further '
                'update '
                'is '
                'anticipated'
            }
        },
        'GROUPTYPE': 'MASTERGROUP',
        'INPUTGRANULE': {
            'INPUTPOINTER': {
                'VALUE':
                ('MYD01.61.2019-051T12:25:00.000000Z.NA.29878844.500100_1.hdf',
                 'MYD03LUT.coeff_V6.1.4',
                 'PM1EPHND_NRT.A2019051.1220.061.2019051125628',
                 'PM1EPHND_NRT.A2019051.1225.061.2019051125628',
                 'PM1EPHND_NRT.A2019051.1230.061.2019051125628', '          '
                 'PM1ATTNR_NRT.A2019051.1220.061.2019051125628',
                 'PM1ATTNR_NRT.A2019051.1225.061.2019051125628',
                 'PM1ATTNR_NRT.A2019051.1230.061.2019051125628')
            }
        },
        'MEASUREDPARAMETER': {
            'MEASUREDPARAMETERCONTAINER': {
                'PARAMETERNAME': {
                    'VALUE': 'Geolocation'
                },
                'QAFLAGS': {
                    'AUTOMATICQUALITYFLAG': {
                        'VALUE': 'Passed'
                    },
                    'AUTOMATICQUALITYFLAGEXPLANATION': {
                        'VALUE':
                        'Set '
                        'to '
                        "'Failed' "
                        'if '
                        'processing '
                        'error '
                        'occurred, '
                        'set '
                        'to '
                        "'Passed' "
                        'otherwise'
                    },
                    'SCIENCEQUALITYFLAG': {
                        'VALUE': 'Not '
                        'Investigated'
                    }
                },
                'QASTATS': {
                    'QAPERCENTMISSINGDATA': {
                        'VALUE': 0
                    },
                    'QAPERCENTOUTOFBOUNDSDATA': {
                        'VALUE': 0
                    }
                }
            }
        },
        'ORBITCALCULATEDSPATIALDOMAIN': {
            'ORBITCALCULATEDSPATIALDOMAINCONTAINER': {
                'EQUATORCROSSINGDATE': {
                    'VALUE': '2019-02-20'
                },
                'EQUATORCROSSINGLONGITUDE': {
                    'VALUE': -151.260740805733
                },
                'EQUATORCROSSINGTIME': {
                    'VALUE': '12:49:52.965727'
                },
                'ORBITNUMBER': {
                    'VALUE': 89393
                }
            }
        },
        'PGEVERSIONCLASS': {
            'PGEVERSION': {
                'VALUE': '6.1.4'
            }
        },
        'RANGEDATETIME': {
            'RANGEBEGINNINGDATE': {
                'VALUE': '2019-02-20'
            },
            'RANGEBEGINNINGTIME': {
                'VALUE': '12:25:00.000000'
            },
            'RANGEENDINGDATE': {
                'VALUE': '2019-02-20'
            },
            'RANGEENDINGTIME': {
                'VALUE': '12:30:00.000000'
            }
        },
        'SPATIALDOMAINCONTAINER': {
            'HORIZONTALSPATIALDOMAINCONTAINER': {
                'GPOLYGON': {
                    'GPOLYGONCONTAINER': {
                        'GRING': {
                            'EXCLUSIONGRINGFLAG': {
                                'VALUE': 'N'
                            }
                        },
                        'GRINGPOINT': {
                            'GRINGPOINTLATITUDE': {
                                'VALUE': (29.5170117594673, 26.1480434828114,
                                          43.2445462598877, 47.7959787025408)
                            },
                            'GRINGPOINTLONGITUDE': {
                                'VALUE': (25.3839329817764, 1.80418778807854,
                                          -6.50842421663422, 23.0260060198343)
                            },
                            'GRINGPOINTSEQUENCENO': {
                                'VALUE': (1, 2, 3, 4)
                            }
                        }
                    }
                }
            }
        }
    }
}

metadata_modisl1b = """
GROUP=SwathStructure
    GROUP=SWATH_1
        SwathName="MODIS_SWATH_Type_L1B"
            GROUP=DimensionMap
            OBJECT=DimensionMap_1
                GeoDimension="2*nscans"
                DataDimension="10*nscans"
                Offset=2
                Increment=5
            END_OBJECT=DimensionMap_1
            OBJECT=DimensionMap_2
                GeoDimension="1KM_geo_dim"
                DataDimension="Max_EV_frames"
                Offset=2
                Increment=5
            END_OBJECT=DimensionMap_2
        END_GROUP=DimensionMap
    END_GROUP=SWATH_1
END_GROUP=SwathStructure
END
"""  # noqa: E501

metadata_modisl2 = """
GROUP=SwathStructure
    GROUP=SWATH_1
        SwathName="mod35"
        GROUP=DimensionMap
            OBJECT=DimensionMap_1
                GeoDimension="Cell_Across_Swath_5km"
                DataDimension="Cell_Across_Swath_1km"
                Offset=2
                Increment=5
            END_OBJECT=DimensionMap_1
            OBJECT=DimensionMap_2
                GeoDimension="Cell_Along_Swath_5km"
                DataDimension="Cell_Along_Swath_1km"
                Offset=2
                Increment=5
            END_OBJECT=DimensionMap_2
        END_GROUP=DimensionMap
        GROUP=IndexDimensionMap
        END_GROUP=IndexDimensionMap
    END_GROUP=SWATH_1
END_GROUP=SwathStructure
END
"""  # noqa: E501


class TestReadMDA(unittest.TestCase):
    """Test reading metadata."""

    def test_read_mda(self):
        """Test reading basic metadata."""
        from satpy.readers.hdfeos_base import HDFEOSBaseFileReader
        res = HDFEOSBaseFileReader.read_mda(nrt_mda)
        self.assertDictEqual(res, nrt_mda_dict)

    def test_read_mda_geo_resolution(self):
        """Test reading geo resolution."""
        from satpy.readers.hdfeos_base import HDFEOSGeoReader
        resolution_l1b = HDFEOSGeoReader.read_geo_resolution(
            HDFEOSGeoReader.read_mda(metadata_modisl1b)
            )
        self.assertEqual(resolution_l1b, 1000)
        resolution_l2 = HDFEOSGeoReader.read_geo_resolution(
            HDFEOSGeoReader.read_mda(metadata_modisl2)
        )
        self.assertEqual(resolution_l2, 5000)
