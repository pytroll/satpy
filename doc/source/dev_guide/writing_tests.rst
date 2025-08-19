==================
Writing unit tests
==================

Satpy tests are written using the third-party :doc:`pytest <pytest:index>`
package.

Fixtures
========

The usage of Pytest `fixtures <https://docs.pytest.org/en/stable/reference/fixtures.html>`_
is encouraged for code re-usability.

As the builtin fixtures (and those defined in ``conftest.py`` file) are injected by
Pytest without them being imported explicitly, their usage can be very confusing for
new developers. To lessen the confusion, it is encouraged to add a note at the
top of the test modules listing all the automatically injected external fixtures
that are used in the module::

    # NOTE:
    # The following fixtures are not defined in this file, but are used and injected by Pytest:
    # - tmp_path
    # - fixture_defined_in_conftest.py

Avoiding Warnings
=================

Satpy tries to avoid all warnings being emitted during tests. A warning
(ex. UserWarning) being emitted during tests suggests there is a new
or upcoming change that will affect Satpy or its users. At the time of
writing Satpy does not fail when warnings are encountered but this may
change in the future.

Warnings encountered during testing should be handled in one of a couple
different ways.

1. Fix the underlying issue. For example, if a dependency is changing behavior
then update Satpy's usage to not produce the warning.
2. Catch the specific warning as part of the test. For example, if a test is
expecting to produce the warning should be making sure that it is, do::

   with pytest.warns(UserWarning, match="the warning message"):
       # code being tested

3. Ignore the error at the test level::

   @pytest.mark.filterwarnings("ignore:the warning message:UserWarning")
   def test_something():
       # test code

4. Ignore the warning globally. This is typically reserved for dependency
changes that are expected to be removed in a future version. These are
configured in the ``pyproject.toml`` in the root of the repository in the
``tool.pytest.ini_options`` section. See existing warning filters there for
examples.

Other tips for avoiding warnings:

* Create semi-realistic test data and avoid ``da.zeros`` or ``da.ones`` when
creating test data. A simple option is to use ``arange``::

  test_data = da.arange(100 * 200).reshape((100, 200)).rechunk(50)

* If using pytest's ``parametrize`` functionality and only some of the
parameters should produce a warning, use ``contextlib.nullcontext``::

  exp_warning = pytest.warns(...) if condition else contextlib.nullcontext()
  with exp_warning:
      # code being tested

Lastly, note that all Numpy "invalid value" warnings are ignored globally in
the ``pyproject.toml`` file. So if your test involves NaNs in the data and
you are expecting to see them or expecting to catch them then you may have
to customize the warning filters with pytest's ``filterwarnings``. See
:ref:`user_warnings_errors` for more details on what is recommended for
users.
