"""Unit tests for Sentinel-5 wavelength handling.

These tests cover discovery and generation of wavelength datasets
derived from Sentinel-5 nominal and calibrated wavelength coefficient
datasets.

The tests verify:

* wavelength dataset discovery
* nominal and calibrated coefficient selection
* wavelength generation from Chebyshev coefficients
* dataset dispatch behaviour
* handling of invalid wavelength coefficient datasets

Mission-independent wavelength utilities are tested separately in
``test_uvns_core.py``.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.readers.uvns import (
    Sentinel5WavelengthHandler,
    VariableRecord,
)


class FakeFileHandler:
    """Minimal file-handler stub used by Sentinel-5 wavelength tests."""

    def __init__(self, records, datasets):
        """Store test records and datasets."""
        self._records = records
        self._datasets = datasets

        self.filetype_info = {
            "file_type": "uvns_l1",
        }

    def __getitem__(self, key):
        """Return a dataset by key."""
        return self._datasets[key]


def make_record(path):
    """Create a minimal VariableRecord for wavelength discovery tests."""
    return VariableRecord(
        path=path,
        name=path.rsplit("/", 1)[-1],
        dataset_name="",
        dimensions=(),
        shape=(),
        dtype="float32",
        attrs={},
    )


def valid_coeffs():
    """Create a valid coefficient dataset for discovery tests."""
    return xr.DataArray(
        da.ones((2, 3, 1), chunks=(2, 3, 1))
    )


def test_discover_nominal_wavelength_products():
    """Verify nominal coefficients generate nominal and best datasets."""
    records = {
        "band1/instrument_data/nominal_wavelength_coefficients":
            make_record(
                "band1/instrument_data/nominal_wavelength_coefficients"
            ),
    }

    coeffs = valid_coeffs()

    fh = FakeFileHandler(
        records,
        {
            "band1/instrument_data/nominal_wavelength_coefficients":
                coeffs,
        },
    )

    handler = Sentinel5WavelengthHandler(fh)

    datasets = handler.discover()

    assert "band1__wavelength" in datasets
    assert "band1__nominal_wavelength" in datasets
    assert "band1__calibrated_wavelength" not in datasets


def test_discover_nominal_and_calibrated_products():
    """Verify nominal and calibrated wavelength datasets are discovered."""
    records = {
        "band1/instrument_data/nominal_wavelength_coefficients":
            make_record(
                "band1/instrument_data/nominal_wavelength_coefficients"
            ),
        "band1/instrument_data/calibrated_wavelength_coefficients":
            make_record(
                "band1/instrument_data/calibrated_wavelength_coefficients"
            ),
    }

    coeffs = valid_coeffs()

    datasets = {
        "band1/instrument_data/nominal_wavelength_coefficients":
            coeffs,
        "band1/instrument_data/calibrated_wavelength_coefficients":
            coeffs,
    }

    fh = FakeFileHandler(
        records,
        datasets,
    )

    handler = Sentinel5WavelengthHandler(fh)

    result = handler.discover()

    assert "band1__wavelength" in result
    assert "band1__nominal_wavelength" in result
    assert "band1__calibrated_wavelength" in result


def test_select_calibrated_coefficients_when_available():
    """Verify calibrated coefficients are preferred when available."""
    records = {
        "band1/instrument_data/calibrated_wavelength_coefficients":
            make_record(
                "band1/instrument_data/calibrated_wavelength_coefficients"
            ),
    }

    coeffs = valid_coeffs()

    fh = FakeFileHandler(
        records,
        {
            "band1/instrument_data/calibrated_wavelength_coefficients":
                coeffs,
        },
    )

    handler = Sentinel5WavelengthHandler(fh)

    coeff_name, source = (
        handler._select_wavelength_coefficients(
            "band1"
        )
    )

    assert coeff_name == (
        "calibrated_wavelength_coefficients"
    )

    assert source == "calibrated"


def test_fallback_to_nominal_coefficients(monkeypatch):
    """Verify coefficient selection falls back to nominal data."""
    records = {
        "band1/instrument_data/nominal_wavelength_coefficients":
            make_record(
                "band1/instrument_data/nominal_wavelength_coefficients"
            ),
    }

    fh = FakeFileHandler(
        records,
        {},
    )

    handler = Sentinel5WavelengthHandler(fh)

    monkeypatch.setattr(
        handler,
        "_coefficients_have_data",
        lambda *args: False,
    )

    coeff_name, source = (
        handler._select_wavelength_coefficients(
            "band1"
        )
    )

    assert coeff_name == (
        "nominal_wavelength_coefficients"
    )

    assert source == "nominal"


def test_generate_nominal_wavelength_dataset():
    """Verify nominal coefficients generate a wavelength dataset."""
    coeffs = xr.DataArray(
        np.array([[[500.0]]]),
    )

    records = {
        "band1/instrument_data/nominal_wavelength_coefficients":
            make_record(
                "band1/instrument_data/nominal_wavelength_coefficients"
            ),
    }

    fh = FakeFileHandler(
        records,
        {
            "band1/instrument_data/nominal_wavelength_coefficients":
                coeffs,
            "band1/spectral_channel":
                np.arange(5),
        },
    )

    handler = Sentinel5WavelengthHandler(
        fh
    )

    ds = handler._get_nominal_wavelength(
        "band1",
        "band1__nominal_wavelength",
    )

    assert ds.shape == (5, 1, 1)

    np.testing.assert_allclose(
        ds.values,
        500.0,
    )

    assert (
        ds.attrs["wavelength_source"]
        == "nominal"
    )


def test_generate_linear_wavelengths():
    """Verify linear Chebyshev coefficients expand correctly."""
    coeffs = np.zeros(
        (1, 1, 2),
        dtype=np.float64,
    )

    coeffs[..., 0] = 500.0
    coeffs[..., 1] = 10.0

    coeffs = xr.DataArray(coeffs)

    records = {
        "band1/instrument_data/nominal_wavelength_coefficients":
            make_record(
                "band1/instrument_data/nominal_wavelength_coefficients"
            ),
    }

    fh = FakeFileHandler(
        records,
        {
            "band1/instrument_data/nominal_wavelength_coefficients":
                coeffs,
            "band1/spectral_channel":
                np.arange(5),
        },
    )

    handler = Sentinel5WavelengthHandler(
        fh
    )

    ds = handler._get_nominal_wavelength(
        "band1",
        "band1__nominal_wavelength",
    )

    expected = (
        500.0
        + 10.0 * np.linspace(-1, 1, 5)
    )

    np.testing.assert_allclose(
        ds[:, 0, 0].values,
        expected,
    )


def test_get_dataset_best_dispatch(monkeypatch):
    """Verify best-wavelength requests are dispatched correctly."""
    handler = Sentinel5WavelengthHandler(
        FakeFileHandler({}, {})
    )

    expected = object()

    called = {}

    def fake_best(
        band_path,
        dataset_name,
    ):
        called["band_path"] = band_path
        called["dataset_name"] = dataset_name
        return expected

    monkeypatch.setattr(
        handler,
        "_get_best_wavelength",
        fake_best,
    )

    ds_info = {
        "name": "band1__wavelength",
        "derived_type": "best",
        "band_path": "band1",
    }

    result = handler.get_dataset(
        ds_info
    )

    assert result is expected

    assert called == {
        "band_path": "band1",
        "dataset_name": "band1__wavelength",
    }


def test_get_dataset_nominal_dispatch(
    monkeypatch,
):
    """Verify nominal-wavelength requests are dispatched correctly."""
    handler = Sentinel5WavelengthHandler(
        FakeFileHandler({}, {})
    )

    expected = object()

    called = {}

    def fake_nominal(
        band_path,
        dataset_name,
    ):
        called["band_path"] = band_path
        called["dataset_name"] = dataset_name
        return expected

    monkeypatch.setattr(
        handler,
        "_get_nominal_wavelength",
        fake_nominal,
    )

    ds_info = {
        "name": "band1__nominal_wavelength",
        "derived_type": "nominal",
        "band_path": "band1",
    }

    result = handler.get_dataset(
        ds_info
    )

    assert result is expected

    assert called == {
        "band_path": "band1",
        "dataset_name":
            "band1__nominal_wavelength",
    }


def test_get_dataset_calibrated_dispatch(
    monkeypatch,
):
    """Verify calibrated-wavelength requests are dispatched correctly."""
    handler = Sentinel5WavelengthHandler(
        FakeFileHandler({}, {})
    )

    expected = object()

    called = {}

    def fake_calibrated(
        band_path,
        dataset_name,
    ):
        called["band_path"] = band_path
        called["dataset_name"] = dataset_name
        return expected

    monkeypatch.setattr(
        handler,
        "_get_calibrated_wavelength",
        fake_calibrated,
    )

    ds_info = {
        "name": "band1__calibrated_wavelength",
        "derived_type": "calibrated",
        "band_path": "band1",
    }

    result = handler.get_dataset(
        ds_info
    )

    assert result is expected

    assert called == {
        "band_path": "band1",
        "dataset_name":
            "band1__calibrated_wavelength",
    }


def test_get_dataset_unknown_type():
    """Verify unsupported derived dataset types raise KeyError."""
    handler = Sentinel5WavelengthHandler(
        FakeFileHandler({}, {})
    )

    with pytest.raises(KeyError):
        handler.get_dataset(
            {
                "name": "bad",
                "derived_type": "unknown",
                "band_path": "band1",
            }
        )


def test_discover_ignores_invalid_nominal_coefficients():
    """Verify invalid nominal coefficients do not generate datasets.

    Wavelength products should only be advertised when coefficient
    datasets contain at least one valid wavelength solution value.
    Datasets containing only invalid values must be ignored during
    discovery.
    """
    records = {
        "band1/instrument_data/nominal_wavelength_coefficients":
            make_record(
                "band1/instrument_data/nominal_wavelength_coefficients"
            ),
    }

    coeffs = xr.DataArray(
        da.full(
            (2, 3, 1),
            np.nan,
            chunks=(2, 3, 1),
        )
    )

    fh = FakeFileHandler(
        records,
        {
            "band1/instrument_data/nominal_wavelength_coefficients":
                coeffs,
        },
    )

    handler = Sentinel5WavelengthHandler(
        fh
    )

    result = handler.discover()

    assert "band1__nominal_wavelength" not in result
    assert "band1__wavelength" not in result


def test_discover_ignores_invalid_calibrated_coefficients():
    """Verify invalid calibrated coefficients do not generate datasets.

    Wavelength products should only be advertised when coefficient
    datasets contain at least one valid wavelength solution value.
    Datasets containing only invalid values must be ignored during
    discovery.
    """
    records = {
        "band1/instrument_data/calibrated_wavelength_coefficients":
            make_record(
                "band1/instrument_data/calibrated_wavelength_coefficients"
            ),
    }

    coeffs = xr.DataArray(
        da.full(
            (2, 3, 1),
            np.nan,
            chunks=(2, 3, 1),
        )
    )

    fh = FakeFileHandler(
        records,
        {
            "band1/instrument_data/calibrated_wavelength_coefficients":
                coeffs,
        },
    )

    handler = Sentinel5WavelengthHandler(
        fh
    )

    result = handler.discover()

    assert (
        "band1__calibrated_wavelength"
        not in result
    )

    assert (
        "band1__wavelength"
        not in result
    )


def test_coefficients_have_data_rejects_invalid_coefficients():
    """Verify invalid coefficient datasets are treated as missing.

    Coefficient datasets containing only invalid values do not provide
    a usable wavelength solution and must be excluded from wavelength
    discovery.
    """
    coeffs = xr.DataArray(
        da.full(
            (2, 3, 1),
            np.nan,
            chunks=(2, 3, 1),
        )
    )

    fh = FakeFileHandler(
        {},
        {
            "band1/instrument_data/nominal_wavelength_coefficients":
                coeffs,
        },
    )

    handler = Sentinel5WavelengthHandler(
        fh
    )

    assert not handler._coefficients_have_data(
        "band1",
        "nominal_wavelength_coefficients",
    )


def test_calibrated_dataset_requires_valid_coefficients():
    """Verify calibrated wavelength generation rejects invalid coefficients.

    Calibrated wavelength datasets should only be generated when
    calibrated coefficients contain valid data. Requests for
    calibrated wavelengths must fail when no valid calibrated
    coefficients are available.
    """
    records = {
        "band1/instrument_data/calibrated_wavelength_coefficients":
            make_record(
                "band1/instrument_data/calibrated_wavelength_coefficients"
            ),
    }

    coeffs = xr.DataArray(
        da.full(
            (2, 3, 1),
            np.nan,
            chunks=(2, 3, 1),
        )
    )

    fh = FakeFileHandler(
        records,
        {
            "band1/instrument_data/calibrated_wavelength_coefficients":
                coeffs,
        },
    )

    handler = Sentinel5WavelengthHandler(
        fh
    )

    with pytest.raises(KeyError):
        handler._get_calibrated_wavelength(
            "band1",
            "band1__calibrated_wavelength",
        )
