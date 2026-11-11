
"""FCI base reader tests package."""

from satpy.readers.core.fci import calculate_area_extent
from satpy.tests.utils import make_dataid


def test_calculate_area_extent():
    """Test function for calculate_area_extent."""
    dataset_id = make_dataid(name="dummy", resolution=2000.0)

    area_dict = {
        "nlines": 5568,
        "ncols": 5568,
        "line_step": dataset_id["resolution"],
        "column_step": dataset_id["resolution"],
    }

    area_extent = calculate_area_extent(area_dict)

    expected = (-5568000.0, 5568000.0, 5568000.0, -5568000.0)

    assert area_extent == expected
