Feature: Image Comparison

  Scenario Outline: Compare generated image with reference image
    Given I have a <composite> reference image file from <satellite> resampled to <area>
    When I generate a new <composite> image file from <satellite> with <reader> for <area> with clipping <clip>
    Then the generated image should be the same as the reference image

    Examples:
      |satellite |composite  | reader | area | clip |
      |Meteosat-12 | cloudtop | fci_l1c_nc | sve | True |
      |Meteosat-12 | night_microphysics | fci_l1c_nc | sve | True |
      |GOES17   |airmass  | abi_l1b | null | null |
      |GOES16   |airmass  | abi_l1b | null | null |
      |GOES16   |ash      | abi_l1b | null | null |
      |GOES17   |ash      | abi_l1b | null | null |
