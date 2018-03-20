Feature: Loading real data in many formats with the same command

    This feature loads real data from disk and generates resampled images.
    This is made as a way to system test satpy.

    To provide test data to this feature, add a directory called `test_data`
    in the current directory. Under this directory, created a directory for
    each data format you want to test, and under this a directory with data
    called data and a directory with reference images called `ref`, eg:

    test_data
    |_ hrit_msg
    |  |_ data
    |  |  |_ [all the MSG data files]
    |  |_ ref
    |     |_ overview_eurol.png
    |     |_ ...
    |_ viirs_sdr
    |  |_ data
    |  |  |_ [all the viirs SDR files]
    |  |_ ref
    |     |_ true_color_eurol.png
    |     |_ ...
    ...

    @wip
    Scenario Outline: Reading and processing of real data
        Given <format> data is available
        When the user loads the <composite> composite
        And the user resamples the data to <area>
        And the user saves the composite to disk
        Then the resulting image should match the reference image


    Examples: AAPP L1 data
        | format         | composite         | area    |
        | aapp_l1b       | overview          | eurol   |

    Examples: ABI L1 data
        | format         | composite         | area          |
        | abi_l1b        | overview          | -             |
        | abi_l1b        | airmass           | -             |
        | abi_l1b        | natural           | -             |

    # Examples: ACSPO data
    #     | format       | composite         | area          |
    #     | acspo        | overview          | -             |
    #     | acspo        | true_color        | -             |
    #     | acspo        | true_color        | north_america |

    Examples: AHI L1 data
        | format         | composite         | area          |
        | ahi_hsd        | overview          | -             |
        | ahi_hsd        | true_color        | -             |
        | ahi_hsd        | true_color        | australia     |

    Examples: AMSR2 L1 data
        | format         | composite | area |
        | amsr2_l1b      | ice       | moll |

    Examples: CLAVR-X data
        | format         | composite | area |
        | clavrx         | cloudtype | usa  |

    Examples: EPS L1 data
        | format         | composite         | area    |
        | epsl1b         | overview          | eurol   |

    Examples: FCI FDHSI data
        | format         | composite  | area    |
        | fci_fdhsi      | overview   | eurol   |
        | fci_fdhsi      | cloudtop   | eurol   |
        | fci_fdhsi      | true_color | eurol   |

    Examples: GAC data
        | format         | composite  | area    |
        | gac_lac_l1     | overview   | eurol   |
        | gac_lac_l1     | cloudtop   | eurol   |

    # Examples: Generic Images

    # Examples: GEOCAT data
    #     | format       | composite         | area          |
    #     | geocat       | overview          | -             |
    #     | geocat       | true_color        | -             |
    #     | geocat       | true_color        | north_america |

    # Examples: GHRSST OSISAF data
    #     | format         | composite         | area          |
    #     | ghrsst_osisaf  | overview          | -             |
    #     | ghrsst_osisaf | true_color        | -             |
    #     | ghrsst_osisaf | true_color        | north_america |

    # Examples: Caliop v3 data
    #     | format         | composite         | area          |
    #     | hdf4_caliopv3  | overview          | -             |
    #     | hdf4_caliopv3  | true_color        | -             |
    #     | hdf4_caliopv3  | true_color        | north_america |

    Examples: MODIS HDF4-EOS data
        | format         | composite         | area    |
        | hdfeos_l1b     | overview          | eurol   |
        | hdfeos_l1b     | true_color_lowres | eurol   |
        | hdfeos_l1b     | true_color        | eurol   |

    Examples: Electro-L N2 HRIT data
        | format         | composite  | area    |
        | hrit_electrol  | overview   | india   |
        | hrit_electrol  | cloudtop   | india   |

    Examples: GOES HRIT data
        | format         | composite  | area    |
        | hrit_goes      | overview   | usa     |
        | hrit_goes      | cloudtop   | usa     |

    Examples: Himawari HRIT data
        | format         | composite  | area        |
        | hrit_jma       | overview   | australia   |
        | hrit_jma       | cloudtop   | australia   |

    Examples: MSG HRIT data
        | format         | composite  | area    |
        | hrit_msg       | overview   | eurol   |
        | hrit_msg       | cloudtop   | eurol   |

    Examples: HRPT data
        | format         | composite  | area    |
        | hrpt           | overview   | eurol   |
        | hrpt           | cloudtop   | eurol   |

    # Examples: IASI L2 data

    # Examples: Lightning Imager L2

    # Examples: MAIA data

    Examples: MSG Native data
        | format         | composite  | area    |
        | native_msg     | overview   | eurol   |
        | native_msg     | cloudtop   | eurol   |

    Examples: NWCSAF GEO data
        | format         | composite  | area    |
        | nc_nwcsaf_msg  | cloudtype  | eurol   |
        | nc_nwcsaf_msg  | ctth       | eurol   |

    Examples: NWCSAF PPS data
        | format         | composite  | area    |
        | nc_nwcsaf_pps  | cloudtype  | eurol   |
        | nc_nwcsaf_pps  | ctth       | eurol   |

    Examples: MSG Native data
        | format         | composite  | area    |
        | native_msg     | overview   | eurol   |
        | native_msg     | cloudtop   | eurol   |

    Examples: OLCI L1 data
        | format         | composite         | area    |
        | nc_olci_l1b    | true_color        | eurol   |

    Examples: OLCI L2 data
        | format         | composite         | area    |
        | nc_olci_l2     | karo              | eurol   |

    Examples: SLSTR L1 data
        | format         | composite         | area    |
        | nc_slstr       | true_color        | eurol   |

    # Examples: NUCAPS data

    # Examples: OMPS EDR

    Examples: SAFE MSI L1 data
        | format         | composite         | area    |
        | safe_msi       | true_color        | eurol   |

    Examples: SAR-C L1 data
        | format         | composite | area   |
        | safe_sar_c     | sar-ice   | euron1 |
        | safe_sar_c     | sar-rgb   | euron1 |
        | safe_sar_c     | sar-quick | euron1 |

    # Examples: SCATSAT 1 data
    #     | format         | composite | area  |
    #     | sar_c          | ice       | eurol |

    Examples: VIIRS compact data
        | format         | composite  | area    |
        | viirs_compact  | overview   | eurol   |
        | viirs_compact  | true_color | eurol   |

    Examples: VIIRS L1B data
        | format         | composite  | area    |
        | viirs_l1b      | overview   | eurol   |
        | viirs_l1b      | true_color | eurol   |

    Examples: VIIRS SDR data
        | format         | composite          | area    |
        | viirs_sdr      | overview           | eurol   |
        | viirs_sdr      | true_color_lowres  | eurol   |
        | viirs_sdr      | fog                | eurol   |
        | viirs_sdr      | dust               | eurol   |
        | viirs_sdr      | ash                | eurol   |
        | viirs_sdr      | natural_sun_lowres | eurol   |
        | viirs_sdr      | snow_age           | eurol   |
