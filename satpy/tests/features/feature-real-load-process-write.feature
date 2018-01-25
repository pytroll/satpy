Feature: Loading real data in many formats with the same command

    This feature loads real data from disk and generates resampled images.
    This is made as a way to system test satpy.

    @wip
    Scenario Outline: Reading and processing of real data
        Given <format> data is available
        When the user loads the <composite> composite
        And the user resamples the data to <area>
        And the user saves the composite to disk
        Then the resulting image should match the reference image


    Examples: MSG HRIT data
        | format         | composite  | area    |
        | hrit_msg       | overview   | eurol   |
        | hrit_msg       | cloudtop   | eurol   |

    Examples: MSG Native data
        | format         | composite  | area    |
        | native_msg     | overview   | eurol   |
        | native_msg     | cloudtop   | eurol   |

    Examples: VIIRS SDR data
        | format         | composite  | area    |
        | viirs_sdr      | overview   | eurol   |
        | viirs_sdr      | true_color | eurol   |
