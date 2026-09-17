# Satpy — Agent Guide

Satpy is a [Pytroll](https://pytroll.github.io/) library for reading, manipulating, and writing data
from remote-sensing earth-observing satellite instruments.
Satpy is an abstraction layer over `pyresample`, `pyspectral`, `trollimage`, `pycoast`, and
`python-geotiepoints` libraries.

If something below is not clear, consult the Sphinx documentation in `doc/source/`; the
"Where to read more" table at the bottom points at the most relevant pages.

## Core concepts

Satpy uses a high-level `Scene` object (`satpy/scene.py`) to wrap the functionality of the other parts
of Satpy. The Scene is both a container for the data being worked with and the interface to act on
that data. Every component of Satpy is technically optional: the component classes can be imported
and called directly, and users can skip anything their task doesn't need. A `Scene` is not required,
it is only the easiest way to tie the components together.

Data processed by Satpy may be referred to as a dataset, a product, channel, or a band.
Data is always an `xarray.DataArray` wrapping a **dask array**. Dims should be `y` and `x`.
A third `bands` dimension is common for representing image bands (ex. R, G, B). Other dimensions
(ex. time) are also possible. In rare cases 1D arrays are used and may only have a `y` dimension.
Metadata lives in `.attrs`; `area`, `start_time`, `end_time`, `units`, `standard_name`, and `sensor`
are expected to be present.
A `DataID` (`satpy/dataset/dataid.py`) object is used as the identifier for each product.
A user may use a `DataQuery` (`satpy/dataset/dataid.py`) to access a product of a specific `DataID`
as shorthand for the full `DataID` or when not all parts of the ID are known.
Default ID keys for imager bands: `name`, `wavelength`, `resolution`, `calibration`, `modifiers`.

### Geographic data

Geolocation objects are stored in the `.attrs["area"]` field. The types of geographic data Satpy
deals with:

- 2D projected data: Data mapped to a projected grid of pixels. Dimensions are `y` and `x`
  and are most often in units of "meters" or "degrees". Geolocation is usually defined by
  an `AreaDefinition` object from `pyresample` using a Coordinate Reference System (CRS) to define
  the projection, width and height to define the shape in pixels, and extents to define the outer
  edges of the area in projection units.
- 2D swath data: Data recorded from a polar-orbiting satellite where the `y` dimension usually is the
  along-track direction and `x` is the cross-track direction. Geolocation is usually defined by
  a 2D longitude and 2D latitude array contained in a `SwathDefinition` object from `pyresample`.
- 1D data: Sometimes 1-dimensional data is processed that represents an unstructured version of one
  of the above 2 cases or some other case not well represented by the existing structures.

### High-level components

The main components wrapped by the `Scene` and used in a typical user workflow are:

- **Readers**: Given input data files (typically on-disk) determine which files can be read and make
  the contents available to the user as `xarray.DataArray` objects identified by a `DataID`. Readers
  use a collection of file handlers (one per input file) to do the actual reading of the files.
- **Compositors**: Combine multiple datasets to create a new dataset. It is common for a composite's
  `standard_name` to be used to link it to an enhancement configuration.
- **Modifiers**: Transform or correct a single band. Generally modifiers retain the overall meaning
  of the original data, but have adjusted some aspect of it (ex. removing atmospheric effects).
- **Writers**: Write data to another (typically on-disk) format. Writers also use "enhancements" to scale
  data to be written.

All high-level components are a Python class plus a YAML config. YAML under `satpy/etc/` points at
classes with `!!python/name:` and is loaded with PyYAML's `UnsafeLoader` (which is why `.bandit`
skips B506). Satpy typically finds a YAML configuration and then loads the associated
Python object with information from the YAML file. The YAML is what makes a component
*discoverable*: a class with no YAML entry can still be imported and called directly, but nothing
config-driven (`Scene.load`, `available_dataset_names`, the `DependencyTree`) will ever find it.

When composites or modifiers have dependencies the `DependencyTree` (`satpy/dependency_tree.py`)
is used to resolve which dataset should be used and where it should come from
(ex. another composite versus a loaded reader).

### Low-level components

- **Resamplers**: Remap data from one geolocation to another (swath → area, area → area, …). Usually
  referenced by a short string name rather than instantiated directly.
- **Enhancements**: Normalize (0-1) or colorize data (ex. single band -> RGB) before data is written
  to an output format or visualized (ex. matplotlib plot).

### Typical workflow

`Scene(filenames, reader=)` → `load([...])` → `resample(area)` → `save_datasets()`

Internally that is **readers** → **compositors** / **modifiers** (ordered by the `DependencyTree`) →
**resampling** → **enhancements** → **writers**.

## MultiScene

The `MultiScene` (`satpy/multiscene/_multiscene.py`, still experimental) was created to make working
with multiple `Scene` objects easier. Some features include the ability to save animations by
treating each Scene as a frame or combining multiple Scenes into a single dataset/image.

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
├── etc/                  YAML config: readers/ composites/ enhancements/ writers/ areas.yaml
└── tests/                mirrors the source split (see Testing below)
```

## Where things live

| Thing | Path |
|---|---|
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
- **Stay lazy.** Avoid `.compute()`, `.values`, `np.asarray()`, or `bool()` on data in library
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
pytest satpy/tests                                    # unit tests
pytest satpy/tests/reader_tests/test_abi_l1b.py       # single module
cd doc && make html                                   # docs; RTD builds with fail_on_warning
asv run                                               # benchmarks (benchmarks/, not in CI)

# Only when explicitly asked -- never part of routine verification:
behave satpy/tests/features --tags=-download          # BDD tests; also run in CI
behave satpy/tests/behave/features                    # reference-image comparison
```

CI runs the unit tests and behave tests on Linux/macOS/Windows × Python 3.11/3.12/3.13. Linting is
delegated to pre-commit.ci. Env setup: `doc/source/dev_guide/index.rst`.

**Default verification is `pytest satpy/tests`.** Do not run the behave or image-comparison suites
unless explicitly asked. `satpy/tests/features/` (behave BDD) runs in CI; `satpy/tests/behave/`
(reference-image comparison) is not wired into CI and additionally expects reference data at a
hardcoded `/app/ext_data` (`satpy/tests/behave/features/steps/image_comparison.py`), so it will not
work on a normal dev machine without that data.

## Style and linting

- ruff, `line-length = 120`. Google-convention docstrings on all public modules, classes, and
  functions. Full rule set and isort config live in `pyproject.toml`.
- There is **no `ruff-format`** — do not reformat code you did not otherwise touch.

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
