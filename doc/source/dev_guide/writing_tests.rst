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
