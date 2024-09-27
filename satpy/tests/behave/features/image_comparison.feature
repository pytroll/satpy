Feature: Image Comparison

  Scenario Outline: Compare generated image with reference image
    Given I have a <composite> reference image file from <satellite>
    When I generate a new <composite> image file from <satellite>
    Then the generated image should be the same as the reference image

    Examples:
      |satellite |composite  |
      |GOES17   |airmass  |
      |GOES16   |airmass  |
      |GOES16   |ash      |
      |GOES17   |ash      |