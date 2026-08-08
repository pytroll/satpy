# Satpy — Agent Guide

Satpy is a [Pytroll](https://pytroll.github.io/) library for reading, manipulating, and writing data
from remote-sensing earth-observing satellite instruments.
Satpy is an abstraction layer over `pyresample`, `pyspectral`, `trollimage`, `pycoast`, and
`python-geotiepoints`.

## Core concepts

- `Scene` (`satpy/scene.py`) is the user-facing container. Typical flow: `Scene(filenames, reader=)`
  → `load([...])` → `resample(area)` → `save_datasets()`.
- Data is always an `xarray.DataArray` wrapping a **dask array**. Dims are `y`, `x`, and `bands`.
  Metadata lives in `.attrs`; `area`, `start_time`, `end_time`, and `sensor` are expected to be
  present.
- `DataID` / `DataQuery` (`satpy/dataset/dataid.py`) are the keys in a Scene's `DatasetDict`.
  Default ID keys for imager bands: `name`, `wavelength`, `resolution`, `calibration`, `modifiers`.
- Pipeline: **readers** → **composites** (combine bands) → **modifiers** (transform one band) →
  **resampling** → **enhancements** (stretch to 0–1) → **writers**. `DependencyTree`
  (`satpy/dependency_tree.py`) resolves composite/modifier prerequisites.
- **Every component is a Python class *plus* a YAML config.** YAML under `satpy/etc/` points at
  classes with `!!python/name:` and is loaded with PyYAML's `UnsafeLoader` (which is why `.bandit`
  skips B506). Adding a class without a matching YAML entry does nothing.
- A composite's `standard_name` is what links it to its enhancement rule.

## Repo layout

```
satpy/
├── scene.py              Scene — the main user API
├── _config.py            donfig config object, config_search_paths(), SATPY_CONFIG_PATH handling
├── dependency_tree.py    composite/modifier dependency resolution
├── dataset/              DataID, DataQuery, DatasetDict, combine_metadata
├── readers/              ~125 reader modules; readers/core/ holds the framework + format bases
├── writers/              geotiff, cf_writer, awips_tiled, …; writers/core/ holds Writer/ImageWriter
├── composites/           core.py + arithmetic/fill/mask/lookup/resolution/spectral + per-sensor
├── enhancements/         enhancer.py + contrast/colormap/wrappers/convolution/overlays
├── modifiers/            base.py + geometry/atmosphere/spectral/angles/parallax/filters
├── resample/             base.py dispatch + kdtree/native/bucket/ewa
├── cf/                   CF-NetCDF encoding used by writers/cf_writer.py
├── multiscene/           MultiScene (experimental) + blend_funcs
├── demo/                 demo data downloaders
├── etc/                  YAML config: readers/ composites/ enhancements/ writers/ areas.yaml
└── tests/                mirrors the source split (see Testing below)
```

## Where things live

| Thing | Path |
|---|---|
| `Scene` | `satpy/scene.py` |
| `BaseFileHandler` | `satpy/readers/core/file_handlers.py` |
| `FileYAMLReader` | `satpy/readers/core/yaml_reader.py` |
| Format bases (`NetCDF4FileHandler`, `HDF5FileHandler`, …) | `satpy/readers/core/{netcdf,hdf5,hdf4,hdfeos,hrit}.py` |
| `Writer` / `ImageWriter` | `satpy/writers/core/{base,image}.py` |
| `CompositeBase` / `GenericCompositor` | `satpy/composites/core.py` |
| `ModifierBase` | `satpy/modifiers/base.py` |
| `Enhancer` / `get_enhanced_image` | `satpy/enhancements/enhancer.py` |
| Resamplers | `satpy/resample/{base,kdtree,native,bucket,ewa}.py` |
| Test helpers (`make_dataid`, `assert_maximum_dask_computes`) | `satpy/tests/utils.py` |

## Gotchas and anti-patterns

- **Deprecation shims.** `satpy/readers/__init__.py`, `writers/__init__.py`,
  `composites/__init__.py`, `enhancements/__init__.py`, `multiscene/__init__.py`, and ~20 legacy
  reader modules (`netcdf_utils.py`, `hdf5_utils.py`, `seviri_base.py`, `abi_base.py`, …) are
  module-level `__getattr__` forwarders driven by `satpy.utils._import_and_warn_new_location`.
  **Always import from the real `*.core.*` / split modules.** Old paths are removed in Satpy 1.0.
- `satpy.writers.show` and `satpy.writers.to_image` are hard-removed and raise `AttributeError`.
- **Stay lazy.** Never call `.compute()`, `.values`, `np.asarray()`, or `bool()` on data in library
  code.
- **Do not build dask arrays out of delayed objects.** To wrap a non-dask-friendly function, use
  `da.map_blocks` / `da.map_overlap` / `da.blockwise`, not `dask.delayed` + `da.from_delayed`.
  Modern dask pays a real performance penalty for mixing delayed objects with dask arrays, and Satpy
  is migrating away from the pattern; the few remaining uses (`satpy/readers/eps_l1b.py`,
  `aapp_l1b.py`, `readers/core/hrit.py`, `writers/mitiff.py`) are legacy, not examples to copy.
  This does **not** apply to `Delayed` as a *return* type from `save_datasets(compute=False)`, which
  is the intended writer API (see `satpy/writers/core/compute.py`).
- Prefer `np.float32`. Use `.where(cond, np.float32(np.nan))` to avoid silent float64 upcasting.
  NaN is the mask for floats; use a `_FillValue` attr for ints. Cast numpy scalar types in `.attrs`
  to Python builtins (needed for hashing and serialization).
- `"x" in scene` works; `"x" in scene.keys()` is always `False`.
- Chunk size comes from dask's `array.chunk-size` (default 128MiB) via
  `satpy.utils.get_chunk_size_limit` — never hardcode one. `PYTROLL_CHUNK_SIZE` is deprecated.
- Enhancement functions take `img` (a `trollimage.xrimage.XRImage`) and currently mutate `img.data`
  in place, although they *should* return `img`. Build them from the decorators in
  `satpy/enhancements/wrappers.py` (`exclude_alpha`, `on_separate_bands`, `on_dask_array`,
  `using_map_blocks`). An unmatched enhancement silently falls back to a linear 0.5%/99.5% stretch —
  use `standard_name: image_ready` or an empty `operations: []` for a genuine no-op.
- Reader renames live in `satpy/readers/core/config.py`: `PENDING_OLD_READER_NAMES` warns,
  `OLD_READER_NAMES` raises `ValueError`.
- `satpy/version.py` is generated by hatch-vcs — never edit it.
- Docs can lag the code after module moves. If a documented import path fails, check for a
  `*.core.*` equivalent and fix the doc rather than working around it.

## Conventions

- Reader naming: `<sensor>[_<processing level>[_<level detail>]][_<file format>]`, all lowercase,
  underscores *between* fields and hyphens *within* a field (`goes-imager`). The YAML filename stem
  must equal `reader.name`, and `sensors:` entries are all lowercase.
- Reader YAML `status:` is one of `Nominal`, `Beta`, `Alpha`, `Defunct`. Default a **new** reader to
  `Beta` unless there is a specific reason to choose otherwise.
- Composites: `satpy/etc/composites/visir.yaml` is generic, `<sensor>.yaml` is sensor-specific, and
  inheritance is declared via `sensor_name: visir/seviri`.
- Enhancements: `generic.yaml` is always loaded first, then `<sensor>.yaml` is layered on top.
- `platform_name` and `sensor` should follow WMO OSCAR names (https://space.oscar.wmo.int/).
- `SATPY_CONFIG_PATH` adds user config dirs; builtin configs are always included.

## Development

```bash
conda create -n satpy-dev python=3.11 && conda activate satpy-dev
conda install --only-deps satpy && conda install eccodes
pip install -e ".[dev]"          # dev = satpy[doc,tests]
pre-commit install

pytest satpy/tests                                    # unit tests
pytest satpy/tests/reader_tests/test_abi_l1b.py       # single module
cd doc && make html                                   # docs; RTD builds with fail_on_warning
asv run                                               # benchmarks (benchmarks/, not in CI)

# Only when explicitly asked -- never part of routine verification:
behave satpy/tests/features --tags=-download          # BDD tests; also run in CI
behave satpy/tests/behave/features                    # reference-image comparison
```

CI runs the unit tests and behave tests on Linux/macOS/Windows × Python 3.11/3.12/3.13. Linting is
delegated to pre-commit.ci.

**Default verification is `pytest satpy/tests`.** Do not run the behave or image-comparison suites
unless explicitly asked. `satpy/tests/features/` (behave BDD) runs in CI; `satpy/tests/behave/`
(reference-image comparison) is not wired into CI and additionally expects reference data at a
hardcoded `/app/ext_data` (`satpy/tests/behave/features/steps/image_comparison.py`), so it will not
work on a normal dev machine without that data.

## Style and linting

- ruff with `line-length = 120`. (`doc/source/dev_guide/CONTRIBUTING.rst` still says 80 characters;
  120 is what the tooling actually enforces.)
- Rules enabled: `A D E W F I PT TID C90 Q T10 T20 NPY`. Google-convention docstrings are required
  on all public modules, classes, and functions. mccabe max-complexity 10. Double quotes. No `print`.
- isort with the black profile, `known_first_party = "satpy"`.
- pre-commit runs ruff-check, trailing-whitespace, end-of-file-fixer, check-yaml `--unsafe`, bandit,
  mypy, and isort. There is **no `ruff-format`** — do not reformat code you did not otherwise touch.

## Testing

- pytest, not `unittest`. A `TestCase` → pytest migration is in progress (~43 of 122 reader test
  files still use `unittest.TestCase`). Write new tests as plain functions with fixtures; if a class
  is warranted, use a bare class with `setup_method`/`teardown_method`, not `TestCase`.
- Preferred reader-test pattern: write a real file to `tmp_path`, then load it through the real
  reader via `load_readers([...], "<reader_name>")`. `satpy/tests/reader_tests/test_abi_l1b.py` is
  the model. The legacy alternative is the fake file-handler swap-in (`FakeNetCDF4FileHandler` in
  `test_netcdf_utils.py`, `FakeHDF5FileHandler` in `test_hdf5_utils.py`, patched via
  `mock.patch.object(Handler, "__bases__", ...)`).
- Autouse fixtures in `satpy/tests/conftest.py`: `_reset_satpy_config`, `_clear_function_caches`,
  `_forbid_pyspectral_downloads`. Opt into `include_test_etc` to point config at `satpy/tests/etc`.
- Helpers in `satpy/tests/utils.py`: `make_dataid`, `make_dsq`, `FakeCompositor`, `CustomScheduler`,
  `assert_maximum_dask_computes`, `make_fake_scene`. Enhancement tests assert **zero** dask computes
  via `satpy/tests/enhancement_tests/utils.py`.
- `--strict-markers`, `--strict-config`, and `xfail_strict` are on; no custom markers are registered.
- Tests must not emit new warnings. Escalate in this order: fix the underlying cause →
  `pytest.warns(match=...)` → `@pytest.mark.filterwarnings` → global `filterwarnings` in
  `pyproject.toml`.
- Prefer semi-realistic fake data (`da.arange(...).reshape(...).rechunk(...)`) over `zeros`/`ones`.
- Add a `# NOTE:` block at the top of a test module listing externally injected fixtures.
- When moving a module or class, add a parametrized "moved" test —
  `satpy/tests/compositor_tests/test_moved_compositors.py` is the template.

## Where to read more

| Task | Doc |
|---|---|
| Add a reader | `doc/source/dev_guide/custom_reader.rst` (the most detailed dev doc) |
| Add a composite or modifier | `doc/source/composites.rst`, `doc/source/modifiers.rst` |
| Add an enhancement | `doc/source/enhancements.rst` |
| dask / xarray rules | `doc/source/dev_guide/xarray_migration.rst` |
| Write tests | `doc/source/dev_guide/writing_tests.rst`, `doc/source/dev_guide/testing.rst` |
| DataID internals | `doc/source/dev_guide/satpy_internals.rst` |
| Auxiliary data download | `doc/source/dev_guide/aux_data.rst` |
| fsspec / remote files | `doc/source/dev_guide/remote_file_support.rst` |
| Third-party plugins | `doc/source/dev_guide/plugins.rst` (experimental) |
| Config and env vars | `doc/source/config.rst` |
| Releasing | `RELEASING.md` |

There is **no dev-guide doc for adding a writer** — read `satpy/writers/core/base.py` and
`satpy/writers/simple_image.py` instead.

There are also no curated "copy this one" reference implementations for readers, composites, or
enhancements yet — the docs above are the best available starting point. Maintainers intend to add
such examples in the future.
