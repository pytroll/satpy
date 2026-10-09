"""Unit tests for core UVNS reader utilities.

These tests exercise mission-independent reader infrastructure shared
across the UVNS reader implementation, including dataset naming,
attribute normalization, coordinate resolution, wavelength derivation
helpers, and dimension normalization.

Mission-specific wavelength behaviour is tested separately in the
Sentinel-4 and Sentinel-5 test modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from satpy.readers.uvns import (
    AttributeNormalizer,
    CoordinateResolver,
    DatasetDerivations,
    DatasetNameRegistry,
    UVNSFileHandler,
    VariableRecord,
)

# -----------------------------------------------------------------------------
# DatasetNameRegistry
# -----------------------------------------------------------------------------


def test_dataset_name_registry_preserves_unique_basenames():
    """Verify unique basenames are used unchanged as dataset names."""
    registry = DatasetNameRegistry(
        (
            "group1/radiance",
            "group2/geolocation",
        )
    )

    assert registry.dataset_name("group1/radiance") == "radiance"
    assert registry.dataset_name("group2/geolocation") == "geolocation"


def test_dataset_name_registry_disambiguates_duplicates():
    """Verify duplicate basenames are disambiguated using path components."""
    registry = DatasetNameRegistry(
        (
            "band1/radiance",
            "band2/radiance",
        )
    )

    assert registry.dataset_name("band1/radiance") == "band1__radiance"
    assert registry.dataset_name("band2/radiance") == "band2__radiance"


def test_dataset_name_registry_shortest_unique_suffix():
    """Verify the shortest unique suffix is used for duplicate paths."""
    registry = DatasetNameRegistry(
        (
            "a/b/radiance",
            "c/b/radiance",
            "c/d/radiance",
        )
    )

    assert registry.dataset_name("a/b/radiance") == "a__b__radiance"
    assert registry.dataset_name("c/b/radiance") == "c__b__radiance"
    assert registry.dataset_name("c/d/radiance") == "d__radiance"


def test_dataset_name_registry_roundtrip_lookup():
    """Verify dataset names can be resolved back to their variable path."""
    registry = DatasetNameRegistry(
        (
            "group/radiance",
        )
    )

    assert (
        registry.variable_path("radiance")
        == "group/radiance"
    )


# -----------------------------------------------------------------------------
# AttributeNormalizer
# -----------------------------------------------------------------------------


def test_normalise_string_removes_trailing_nul():
    """Verify trailing NUL characters are removed from strings."""
    assert (
        AttributeNormalizer.normalise_value(
            "hello\x00"
        )
        == "hello"
    )


def test_normalise_bytes_to_string():
    """Verify byte strings are decoded and cleaned."""
    assert (
        AttributeNormalizer.normalise_value(
            b"hello\x00"
        )
        == "hello"
    )


def test_normalise_numpy_bytes_array():
    """Verify NumPy byte arrays are converted to string arrays."""
    data = np.array(
        [b"a\x00", b"b\x00"],
        dtype="S2",
    )

    result = AttributeNormalizer.normalise_value(data)

    np.testing.assert_array_equal(
        result,
        np.array(["a", "b"]),
    )


def test_normalise_nested_structures():
    """Verify normalization is applied recursively to nested values."""
    value = (
        b"abc\x00",
        [b"def\x00"],
    )

    result = AttributeNormalizer.normalise_value(value)

    assert result == (
        "abc",
        ["def"],
    )


# -----------------------------------------------------------------------------
# CoordinateResolver
# -----------------------------------------------------------------------------

def make_record(
    path,
    *,
    dimensions=("y", "x"),
    shape=(10, 20),
    attrs=None,
):
    """Create a minimal VariableRecord for coordinate-resolution tests."""
    return VariableRecord(
        path=path,
        name=path.rsplit("/", 1)[-1],
        dataset_name=path.replace("/", "__"),
        dimensions=dimensions,
        shape=shape,
        dtype="float32",
        attrs=attrs or {},
    )


def test_geographic_role_standard_name_latitude():
    """Verify latitude standard names are recognized."""
    role = CoordinateResolver.geographic_role(
        {"standard_name": "latitude"}
    )

    assert role == "latitude"


def test_geographic_role_standard_name_longitude():
    """Verify longitude standard names are recognized."""
    role = CoordinateResolver.geographic_role(
        {"standard_name": "longitude"}
    )

    assert role == "longitude"


def test_geographic_role_from_units():
    """Verify geographic roles can be inferred from coordinate units."""
    assert (
        CoordinateResolver.geographic_role(
            {"units": "degrees_north"}
        )
        == "latitude"
    )

    assert (
        CoordinateResolver.geographic_role(
            {"units": "degrees_east"}
        )
        == "longitude"
    )


def test_parse_coordinate_paths_from_string():
    """Verify coordinate references are parsed from attribute strings."""
    paths = CoordinateResolver._parse_coordinate_paths(
        "longitude latitude"
    )

    assert paths == (
        "longitude",
        "latitude",
    )


def test_resolve_valid_coordinate_pair():
    """Verify valid longitude and latitude coordinates are resolved."""
    records = {
        "radiance": make_record(
            "radiance",
            attrs={
                "coordinates": "longitude latitude",
            },
        ),
        "longitude": make_record(
            "longitude",
            attrs={
                "standard_name": "longitude",
            },
        ),
        "latitude": make_record(
            "latitude",
            attrs={
                "standard_name": "latitude",
            },
        ),
    }

    names = DatasetNameRegistry(records)

    resolver = CoordinateResolver(
        records,
        names,
    )

    result = resolver.resolve(
        records["radiance"]
    )

    assert result == (
        "longitude",
        "latitude",
    )


def test_resolve_missing_coordinate_returns_empty_tuple():
    """Verify incomplete coordinate pairs are rejected."""
    records = {
        "radiance": make_record(
            "radiance",
            attrs={
                "coordinates": "longitude latitude",
            },
        ),
        "longitude": make_record(
            "longitude",
            attrs={
                "standard_name": "longitude",
            },
        ),
    }

    names = DatasetNameRegistry(records)

    resolver = CoordinateResolver(
        records,
        names,
    )

    assert (
        resolver.resolve(records["radiance"])
        == ()
    )


def test_coordinate_dimension_mismatch_rejected():
    """Verify coordinates with incompatible dimensions are rejected."""
    records = {
        "radiance": make_record(
            "radiance",
            dimensions=("y", "x"),
            shape=(10, 20),
            attrs={
                "coordinates": "longitude latitude",
            },
        ),
        "longitude": make_record(
            "longitude",
            dimensions=("y", "x"),
            shape=(10, 20),
            attrs={
                "standard_name": "longitude",
            },
        ),
        "latitude": make_record(
            "latitude",
            dimensions=("scanline", "pixel"),
            shape=(10, 20),
            attrs={
                "standard_name": "latitude",
            },
        ),
    }

    names = DatasetNameRegistry(records)

    resolver = CoordinateResolver(
        records,
        names,
    )

    assert (
        resolver.resolve(records["radiance"])
        == ()
    )


def test_parse_coordinate_paths_normalises_leading_slashes():
    """Verify leading slashes are removed from coordinate references."""
    paths = CoordinateResolver._parse_coordinate_paths(
        "/longitude /latitude"
    )

    assert paths == (
        "longitude",
        "latitude",
    )


def test_resolve_coordinate_same_group_lookup():
    """Verify coordinates can be resolved from the same group."""
    records = {
        "group/radiance": make_record(
            "group/radiance",
            attrs={
                "coordinates": "longitude latitude",
            },
        ),
        "group/longitude": make_record(
            "group/longitude",
            attrs={
                "standard_name": "longitude",
            },
        ),
        "group/latitude": make_record(
            "group/latitude",
            attrs={
                "standard_name": "latitude",
            },
        ),
    }

    names = DatasetNameRegistry(records)

    resolver = CoordinateResolver(
        records,
        names,
    )

    result = resolver.resolve(
        records["group/radiance"]
    )

    assert result == (
        "longitude",
        "latitude",
    )


def test_resolve_coordinate_unique_basename_lookup():
    """Verify uniquely named coordinates can be resolved by basename."""
    records = {
        "radiance": make_record(
            "radiance",
            attrs={
                "coordinates": "longitude latitude",
            },
        ),
        "geo/longitude": make_record(
            "geo/longitude",
            attrs={
                "standard_name": "longitude",
            },
        ),
        "geo/latitude": make_record(
            "geo/latitude",
            attrs={
                "standard_name": "latitude",
            },
        ),
    }

    names = DatasetNameRegistry(records)

    resolver = CoordinateResolver(
        records,
        names,
    )

    result = resolver.resolve(
        records["radiance"]
    )

    assert result == (
        "longitude",
        "latitude",
    )


def test_resolve_coordinate_ambiguous_basename_rejected():
    """Verify ambiguous coordinate references are rejected."""
    # Coordinate references must resolve uniquely. If more than one
    # variable shares the referenced basename the relationship is
    # considered ambiguous and is rejected.

    records = {
        "radiance": make_record(
            "radiance",
            attrs={
                "coordinates": "longitude latitude",
            },
        ),
        "geo1/longitude": make_record(
            "geo1/longitude",
            attrs={
                "standard_name": "longitude",
            },
        ),
        "geo2/longitude": make_record(
            "geo2/longitude",
            attrs={
                "standard_name": "longitude",
            },
        ),
        "latitude": make_record(
            "latitude",
            attrs={
                "standard_name": "latitude",
            },
        ),
    }

    names = DatasetNameRegistry(records)

    resolver = CoordinateResolver(
        records,
        names,
    )

    assert (
        resolver.resolve(
            records["radiance"]
        )
        == ()
    )


def test_referenced_coordinate_paths_tracked():
    """Verify successfully resolved coordinate paths are tracked."""
    # Coordinate datasets referenced by successful resolutions are
    # tracked so they can later be identified as coordinate-only
    # variables.

    records = {
        "radiance": make_record(
            "radiance",
            attrs={
                "coordinates": "longitude latitude",
            },
        ),
        "longitude": make_record(
            "longitude",
            attrs={
                "standard_name": "longitude",
            },
        ),
        "latitude": make_record(
            "latitude",
            attrs={
                "standard_name": "latitude",
            },
        ),
    }

    names = DatasetNameRegistry(records)

    resolver = CoordinateResolver(
        records,
        names,
    )

    resolver.resolve(
        records["radiance"]
    )

    assert (
        resolver.referenced_coordinate_paths
        == frozenset(
            (
                "longitude",
                "latitude",
            )
        )
    )


# -----------------------------------------------------------------------------
# DatasetDerivations
# -----------------------------------------------------------------------------


def test_expand_wavelengths_constant_polynomial():
    """Verify constant polynomial coefficients expand correctly."""
    coeffs = np.ones(
        (2, 3, 1),
        dtype=np.float64,
    ) * 100.0

    result = DatasetDerivations.expand_wavelengths(
        coeffs,
        num_channels=5,
    )

    assert result.shape == (
        5,
        2,
        3,
    )

    np.testing.assert_allclose(
        result,
        100.0,
    )


def test_expand_wavelengths_linear_polynomial():
    """Verify linear polynomial coefficients expand correctly."""
    coeffs = np.zeros(
        (1, 1, 2),
        dtype=np.float64,
    )

    coeffs[..., 0] = 100.0
    coeffs[..., 1] = 10.0

    result = DatasetDerivations.expand_wavelengths(
        coeffs,
        num_channels=5,
    )

    expected_x = np.linspace(
        -1.0,
        1.0,
        5,
    )

    expected = 100.0 + 10.0 * expected_x

    np.testing.assert_allclose(
        result[:, 0, 0],
        expected,
    )


def test_expand_wavelengths_invalid_rank():
    """Verify invalid coefficient array ranks raise an error."""
    coeffs = np.zeros((10, 10))

    with pytest.raises(
        ValueError,
        match=r"Expected wavelength coefficients with shape",
    ):
        DatasetDerivations.expand_wavelengths(
            coeffs,
            num_channels=5,
        )

# -----------------------------------------------------------------------------
# Dimension normalisation
# -----------------------------------------------------------------------------


def test_normalise_dimensions_renames_scanline_ground_pixel():
    """Verify mission-specific dimensions are normalized to y and x."""
    import xarray as xr

    data = xr.DataArray(
        np.zeros((2, 3)),
        dims=("scanline", "ground_pixel"),
    )

    result = UVNSFileHandler._normalise_dimensions(
        data
    )

    assert result.dims == (
        "y",
        "x",
    )


def test_normalise_dimensions_duplicate_names():
    """Verify duplicate dimension names are made unique.

    Matrix-style datasets may legitimately contain multiple dimensions
    with the same extent. xarray warns that duplicate dimension names
    are not fully supported. The UVNS reader therefore normalises
    duplicate names to distinct identifiers before further processing.
    """
    import xarray as xr

    with pytest.warns(
        UserWarning,
        match="Duplicate dimension names present",
    ):
        data = xr.DataArray(
            np.zeros((3, 3)),
            dims=("x", "x"),
        )

    result = UVNSFileHandler._normalise_dimensions(
        data
    )

    assert result.dims == (
        "x_x",
        "x_y",
    )


# -----------------------------------------------------------------------------
# Dataset availability discovery
# -----------------------------------------------------------------------------

def test_available_datasets_merges_matching_configured_dataset():
    """Verify configured datasets are merged with discovered metadata.

    When a configured dataset references a discovered file_key for the
    current file type, the configured metadata should be merged with the
    dynamically discovered metadata and the file_key normalised.
    """
    fh = object.__new__(UVNSFileHandler)

    record = make_record(
        "group/radiance",
    )

    fh._records = {
        record.path: record,
    }

    dynamic_name = DatasetNameRegistry(
        ["group/radiance"]
    ).dataset_name("group/radiance")

    fh._dataset_infos = {
        dynamic_name: {
            "name": dynamic_name,
            "file_key": record.path,
            "units": "W m-2 sr-1",
        },
    }

    fh._derived_dataset_infos = {}

    fh._name_registry = DatasetNameRegistry(
        ["group/radiance"]
    )

    fh.file_type_matches = (
        lambda file_type: file_type == "uvns"
    )

    configured = [
        (
            True,
            {
                "name": "configured_radiance",
                "file_key": "/group/radiance",
                "file_type": "uvns",
                "resolution": 1000,
            },
        )
    ]

    result = list(
        fh.available_datasets(
            configured
        )
    )

    assert len(result) == 1

    is_available, ds_info = result[0]

    assert is_available is True
    assert ds_info["resolution"] == 1000
    assert ds_info["units"] == "W m-2 sr-1"
    assert ds_info["name"] == "configured_radiance"
    assert ds_info["file_key"] == "group/radiance"


def test_available_datasets_returns_unmatched_dataset_unchanged():
    """Verify unmatched configured datasets are passed through unchanged."""
    fh = object.__new__(UVNSFileHandler)

    fh._records = {}
    fh._dataset_infos = {}
    fh._derived_dataset_infos = {}

    fh.file_type_matches = (
        lambda file_type: False
    )

    configured = [
        (
            False,
            {
                "name": "unknown",
                "file_key": "missing/path",
                "file_type": "other",
            },
        )
    ]

    result = list(
        fh.available_datasets(
            configured
        )
    )

    assert result == configured

def test_available_datasets_returns_unconfigured_discovered_dataset():
    """Verify discovered datasets are advertised automatically."""
    fh = object.__new__(UVNSFileHandler)

    record = make_record(
        "group/radiance",
    )

    fh._records = {
        record.path: record,
    }

    fh._dataset_infos = {
        record.dataset_name: {
            "name": record.dataset_name,
            "file_key": record.path,
        },
    }

    fh._derived_dataset_infos = {}

    result = list(
        fh.available_datasets()
    )

    assert result == [
        (
            True,
            {
                "name": record.dataset_name,
                "file_key": record.path,
            },
        )
    ]

def test_available_datasets_includes_derived_datasets():
    """Verify derived wavelength datasets are advertised."""
    fh = object.__new__(UVNSFileHandler)

    fh._records = {}
    fh._dataset_infos = {}

    fh._derived_dataset_infos = {
        "wavelength": {
            "name": "wavelength",
            "derived_type": "best",
        }
    }

    result = list(
        fh.available_datasets()
    )

    assert result == [
        (
            True,
            {
                "name": "wavelength",
                "derived_type": "best",
            },
        )
    ]
