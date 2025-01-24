Feature: Image Comparison

  Scenario Outline: Compare generated image with reference image
    Given I have a <composite> reference image file from <satellite> resampled to <area>
    When I generate a new <composite> image file from <satellite> case <case> with <reader> for <area> with clipping <clip>
    Then the generated image should be the same as the reference image

    Examples:
      |satellite | case | composite  | reader | area | clip |
      |Meteosat-12 | scan_night | cloudtop | fci_l1c_nc | sve | True |
      |Meteosat-12 | scan_night | night_microphysics | fci_l1c_nc | sve | True |
      |Meteosat-12 | mali_day | essl_colorized_low_level_moisture | fci_l1c_nc | mali | False |
      |Meteosat-12 | spain_day | colorized_low_level_moisture_with_vis06 | fci_l1c_nc,fci_l2_nc | spain | False |
      |GOES17   | americas_night | airmass  | abi_l1b | null | null |
      |GOES16   | americas_night | airmass  | abi_l1b | null | null |
      |GOES16   | americas_night | ash      | abi_l1b | null | null |
      |GOES17   | americas_night | ash      | abi_l1b | null | null |
