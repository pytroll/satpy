"""Unit tests for Sentinel-4 wavelength handling.

These tests cover discovery and generation of wavelength datasets
derived from Sentinel-4 assigned and calibrated spectral maps.

The tests verify:

* wavelength dataset discovery
* wavelength generation from Chebyshev coefficients
* best-wavelength selection
* dataset dispatch behaviour
* handling of spectral maps that contain no wavelength coefficients

Mission-independent wavelength utilities are tested separately in
``test_uvns_core.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from satpy.readers.uvns import (
    Sentinel4WavelengthHandler,
    VariableRecord,
)


class FakeFileHandler:
    """Minimal file-handler stub used by Sentinel-4 wavelength tests."""

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


def test_discover_assigned_products():
    """Verify assigned spectral maps generate assigned and best datasets."""
    records = {
        "data/detector/uvvis/assigned_spectral_map":
            make_record(
                "data/detector/uvvis/assigned_spectral_map"
            ),
    }

    coeffs = xr.DataArray(
        np.ones(
            (1, 551, 1),
            dtype=np.float64,
        )
    )

    fh = FakeFileHandler(
        records,
        {
            "data/detector/uvvis/assigned_spectral_map":
                coeffs,
        },
    )

    handler = Sentinel4WavelengthHandler(fh)

    result = handler.discover()

    assert (
        "data__detector__uvvis__assigned_wavelength"
        in result
    )

    assert (
        "data__detector__uvvis__wavelength"
        in result
    )

    assert (
        "data__detector__uvvis__calibrated_wavelength"
        not in result
    )


def test_discover_assigned_and_calibrated_products():
    """Verify assigned and calibrated spectral maps are both discovered."""
    records = {
        "data/detector/uvvis/assigned_spectral_map":
            make_record(
                "data/detector/uvvis/assigned_spectral_map"
            ),
        "data/detector/uvvis/calibrated_spectral_map":
            make_record(
                "data/detector/uvvis/calibrated_spectral_map"
            ),
    }

    coeffs = xr.DataArray(
        np.ones(
            (1, 551, 1),
            dtype=np.float64,
        )
    )

    fh = FakeFileHandler(
        records,
        {
            "data/detector/uvvis/assigned_spectral_map":
                coeffs,
            "data/detector/uvvis/calibrated_spectral_map":
                coeffs,
        },
    )

    handler = Sentinel4WavelengthHandler(fh)

    result = handler.discover()

    assert (
        "data__detector__uvvis__assigned_wavelength"
        in result
    )

    assert (
        "data__detector__uvvis__calibrated_wavelength"
        in result
    )

    assert (
        "data__detector__uvvis__wavelength"
        in result
    )


def test_chebyshev_axis_handles_reversed_indices():
    """Verify reversed spectral-column indices produce a valid axis."""
    detector = "data/detector/uvvis"

    fh = FakeFileHandler(
        {},
        {
            f"{detector}/column_min_spectral": 684,
            f"{detector}/column_max_spectral": 0,
        },
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    x = handler._chebyshev_axis(
        detector
    )

    assert len(x) == 685

    np.testing.assert_allclose(
        x[0],
        1.0,
    )

    np.testing.assert_allclose(
        x[-1],
        -1.0,
    )


def test_expand_wavelengths_constant_polynomial():
    """Verify constant Chebyshev coefficients expand to a constant wavelength."""
    detector = "data/detector/uvvis"

    coeffs = np.ones(
        (49, 551, 1),
        dtype=np.float64,
    ) * 500.0

    fh = FakeFileHandler(
        {},
        {
            f"{detector}/column_min_spectral": 4,
            f"{detector}/column_max_spectral": 0,
        },
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    wavelengths = (
        handler._expand_wavelengths(
            coeffs,
            detector,
        )
    )

    assert wavelengths.shape == (
        5,
        49,
        551,
    )

    np.testing.assert_allclose(
        wavelengths,
        500.0,
    )


def test_expand_wavelengths_linear_polynomial():
    """Verify linear Chebyshev coefficients expand correctly."""
    detector = "data/detector/uvvis"

    coeffs = np.zeros(
        (1, 1, 2),
        dtype=np.float64,
    )

    coeffs[..., 0] = 500.0
    coeffs[..., 1] = 10.0

    fh = FakeFileHandler(
        {},
        {
            f"{detector}/column_min_spectral": 4,
            f"{detector}/column_max_spectral": 0,
        },
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    wavelengths = (
        handler._expand_wavelengths(
            coeffs,
            detector,
        )
    )

    expected = (
        500.0
        + 10.0 * np.array(
            [1.0, 0.5, 0.0, -0.5, -1.0]
        )
    )

    np.testing.assert_allclose(
        wavelengths[:, 0, 0],
        expected,
    )


def test_best_wavelength_prefers_calibrated():
    """Verify calibrated spectral maps are preferred when available."""
    detector = "data/detector/uvvis"

    records = {
        f"{detector}/calibrated_spectral_map":
            make_record(
                f"{detector}/calibrated_spectral_map"
            ),
    }

    fh = FakeFileHandler(
        records,
        {},
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    assert handler._has_calibrated(
        detector
    )


def test_best_wavelength_uses_calibrated(
    monkeypatch,
):
    """Verify best-wavelength generation prefers calibrated data.

    When a calibrated spectral map exists, the Sentinel-4 wavelength
    handler should use it as the preferred wavelength solution.
    """
    detector = "data/detector/uvvis"

    records = {
        f"{detector}/calibrated_spectral_map":
            make_record(
                f"{detector}/calibrated_spectral_map"
            ),
    }

    handler = Sentinel4WavelengthHandler(
        FakeFileHandler(
            records,
            {},
        )
    )

    expected = object()

    called = {}

    def fake_create_dataset(**kwargs):
        called.update(kwargs)
        return expected

    monkeypatch.setattr(
        handler,
        "_create_dataset",
        fake_create_dataset,
    )

    result = handler._get_best_wavelength(
        detector,
        "test_wavelength",
    )

    assert result is expected

    assert called == {
        "detector_path": detector,
        "coefficient_name":
            "calibrated_spectral_map",
        "dataset_name":
            "test_wavelength",
    }


def test_best_wavelength_falls_back_to_assigned(
    monkeypatch,
):
    """Verify best-wavelength generation falls back to assigned data.

    When no calibrated spectral map exists for a detector, the
    assigned spectral map should be used as the best available
    wavelength solution.
    """
    detector = "data/detector/uvvis"

    handler = Sentinel4WavelengthHandler(
        FakeFileHandler({}, {})
    )

    expected = object()

    called = {}

    def fake_create_dataset(**kwargs):
        called.update(kwargs)
        return expected

    monkeypatch.setattr(
        handler,
        "_create_dataset",
        fake_create_dataset,
    )

    result = handler._get_best_wavelength(
        detector,
        "test_wavelength",
    )

    assert result is expected

    assert called == {
        "detector_path": detector,
        "coefficient_name":
            "assigned_spectral_map",
        "dataset_name":
            "test_wavelength",
    }


def test_get_dataset_best_dispatch(
    monkeypatch,
):
    """Verify best-wavelength requests are dispatched correctly."""
    handler = Sentinel4WavelengthHandler(
        FakeFileHandler({}, {})
    )

    expected = object()

    called = {}

    def fake_best(
        detector_path,
        dataset_name,
    ):
        called["detector_path"] = detector_path
        called["dataset_name"] = dataset_name
        return expected

    monkeypatch.setattr(
        handler,
        "_get_best_wavelength",
        fake_best,
    )

    result = handler.get_dataset(
        {
            "name": "test",
            "derived_type": "best",
            "detector_path": "detector",
        }
    )

    assert result is expected

    assert called == {
        "detector_path": "detector",
        "dataset_name": "test",
    }


def test_get_dataset_assigned_dispatch(
    monkeypatch,
):
    """Verify assigned-wavelength requests are dispatched correctly."""
    handler = Sentinel4WavelengthHandler(
        FakeFileHandler({}, {})
    )

    expected = object()

    def fake_create_dataset(**kwargs):
        return expected

    monkeypatch.setattr(
        handler,
        "_create_dataset",
        fake_create_dataset,
    )

    result = handler.get_dataset(
        {
            "name": "assigned",
            "derived_type": "assigned",
            "detector_path": "detector",
        }
    )

    assert result is expected


def test_get_dataset_calibrated_dispatch(
    monkeypatch,
):
    """Verify calibrated-wavelength requests are dispatched correctly."""
    handler = Sentinel4WavelengthHandler(
        FakeFileHandler({}, {})
    )

    expected = object()

    def fake_create_dataset(**kwargs):
        return expected

    monkeypatch.setattr(
        handler,
        "_create_dataset",
        fake_create_dataset,
    )

    result = handler.get_dataset(
        {
            "name": "calibrated",
            "derived_type": "calibrated",
            "detector_path": "detector",
        }
    )

    assert result is expected


def test_get_dataset_unknown_type():
    """Verify unsupported derived dataset types raise KeyError."""
    handler = Sentinel4WavelengthHandler(
        FakeFileHandler({}, {})
    )

    with pytest.raises(KeyError):
        handler.get_dataset(
            {
                "name": "bad",
                "derived_type": "unknown",
                "detector_path": "detector",
            }
        )


def test_discover_ignores_empty_assigned_spectral_map():
    """Verify spectral maps with no coefficients are not advertised.

    Some products may contain spectral-map datasets whose coefficient
    dimension has length zero. These datasets do not provide a valid
    wavelength solution and must not generate derived wavelength
    products.
    """
    records = {
        "data/detector/uvvis/assigned_spectral_map":
            make_record(
                "data/detector/uvvis/assigned_spectral_map"
            ),
    }

    fh = FakeFileHandler(
        records,
        {
            "data/detector/uvvis/assigned_spectral_map":
                xr.DataArray(
                    np.empty(
                        (1, 551, 0),
                        dtype=np.float64,
                    )
                )
        },
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    result = handler.discover()

    assert (
        "data__detector__uvvis__assigned_wavelength"
        not in result
    )

    assert (
        "data__detector__uvvis__wavelength"
        not in result
    )


def test_discover_ignores_empty_calibrated_spectral_map():
    """Verify calibrated spectral maps with no coefficients are ignored.

    Spectral maps whose coefficient dimension has length zero do not
    provide a valid wavelength solution and must not generate derived
    wavelength products.
    """
    records = {
        "data/detector/uvvis/calibrated_spectral_map":
            make_record(
                "data/detector/uvvis/calibrated_spectral_map"
            ),
    }

    fh = FakeFileHandler(
        records,
        {
            "data/detector/uvvis/calibrated_spectral_map":
                xr.DataArray(
                    np.empty(
                        (1, 551, 0),
                        dtype=np.float64,
                    )
                )
        },
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    result = handler.discover()

    assert (
        "data__detector__uvvis__calibrated_wavelength"
        not in result
    )

    assert (
        "data__detector__uvvis__wavelength"
        not in result
    )


def test_coefficients_have_data_rejects_empty_coefficients():
    """Verify zero-length coefficient dimensions are treated as missing.

    Some products contain spectral maps whose coefficient dimension has
    length zero, indicating that no wavelength solution is available.
    Such datasets must be excluded from wavelength discovery.
    """
    detector = "data/detector/uvvis"

    fh = FakeFileHandler(
        {},
        {
            f"{detector}/assigned_spectral_map":
                xr.DataArray(
                    np.empty(
                        (1, 551, 0),
                        dtype=np.float64,
                    )
                ),
        },
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    assert not handler._has_coefficients(
        detector,
        "assigned_spectral_map",
    )

def test_get_coefficients_rejects_empty_coefficients():
    """Verify empty coefficient datasets raise KeyError.

    Some Sentinel-4 products expose spectral maps whose Chebyshev
    coefficient dimension has length zero, indicating that no
    wavelength solution is available. Attempting to retrieve
    coefficients should therefore fail.
    """
    detector = "data/detector/uvvis"

    fh = FakeFileHandler(
        {},
        {
            f"{detector}/assigned_spectral_map":
                xr.DataArray(
                    np.empty(
                        (1, 551, 0),
                        dtype=np.float64,
                    )
                ),
        },
    )

    handler = Sentinel4WavelengthHandler(
        fh
    )

    with pytest.raises(
        KeyError,
        match="contains no Chebyshev coefficients",
    ):
        handler._get_coefficients(
            detector,
            "assigned_spectral_map",
        )
