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

=============
Imagery tests
=============

Satpy has automated image comparison tests using the third-party `behave <https://behave.readthedocs.io/en/stable/>`_ package.

Image comparison tests will run only if an authorised developer comments
"start behave test" in a GitHub pull request.
Running such tests is recommended for any pull request that may affect image
production.

To add new image comparison tests:

- Add one or more entries to the "Examples" table in ``satpy/tests/behave/features/image_comparison.feature``
- Checkout / update the git repository at ``https://github.com/pytroll/image-comparison-tests``
- If needed, add a new directory for a test case inside ``image-comparison-tests/data/satellite_data/<satellite-name>/<case>``.  Do not commit those data.
- Run the script ``satpy/utils/create_reference.py`` to create the reference images.  Note
  that although it reads the reference data, it does not create all reference images defined in the
  behave tests, only those passed as arguments to the script.
- deploy by running ``ansible-playbook playbooks/deploy_image_comparison.yml --ask-become-pass``.
  You may need to install ansible first.
  You will need access to the virtual machine in the European Weather Cloud for this step.
