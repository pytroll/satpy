=================
How to contribute
=================

Thank you for considering contributing to Satpy! Satpy's development team
is made up of volunteers so any help we can get is very appreciated.

Contributions from users are what keep this community going. We welcome
any contributions including bug reports, documentation fixes or updates,
bug fixes, and feature requests. By contributing to Satpy you are providing
code that everyone can use and benefit from.

The following guidelines will describe how the Satpy project structures
its code contributions from discussion to code to package release.

For more information on contributing to open source projects see
`GitHub's Guide <https://opensource.guide/how-to-contribute/>`_.

What can I do?
==============

- Make sure you have a `GitHub account <https://github.com/signup/free>`_.
- Submit a ticket for your issue, assuming one does not already exist.
- If you're uncomfortable using Git/GitHub, see
  `Learn Git Branching <https://learngitbranching.js.org/>`_ or other
  online tutorials.
- If you are uncomfortable contributing to an open source project see:

  * `How to Contribute to an Open Source Project on GitHub <https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github>`_
    video series
  * Aaron Meurer's `Git Workflow <http://www.asmeurer.com/git-workflow/>`_
  * `How to Contribute to Open Source <https://opensource.guide/how-to-contribute/>`_

- See what `issues <https://github.com/pytroll/satpy/issues/>`_ already
  exist. Issues marked
  `good first issue <https://github.com/pytroll/satpy/labels/good%20first%20issue>`_
  or `help wanted <https://github.com/pytroll/satpy/labels/help%20wanted>`_
  can be good issues to start with.
- Read the :doc:`index` for more details on contributing code.
- `Fork <https://help.github.com/articles/fork-a-repo/>`_ the repository on
  GitHub and install the package in development mode.
- Update the Satpy documentation to make it clearer and more detailed.
- Contribute code to either fix a bug or add functionality and submit a
  `Pull Request <https://help.github.com/articles/creating-a-pull-request/>`_.
- Make an example Jupyter Notebook and add it to the
  `available examples <https://github.com/pytroll/pytroll-examples>`_.

What if I break something?
==========================

Not possible. If something breaks because of your contribution it was our
fault. When you submit your changes to be merged as a GitHub
`Pull Request <https://help.github.com/articles/creating-a-pull-request/>`_
they will be automatically tested and checked against coding style rules.
Before they are merged they are reviewed by at least one maintainer of the
Satpy project. If anything needs updating, we'll let you know.

What is expected?
=================

You can expect the Satpy maintainers to help you. We are all volunteers,
have jobs, and occasionally go on vacations. We will try our best to answer
your questions as soon as possible. We will try our best to understand your
use case and add the features you need. Although we strive to make
Satpy useful for everyone there may be some feature requests that we can't
allow if they would require breaking existing features. Other features may
be best for a different package, PyTroll or otherwise. Regardless, we will
help you find the best place for your feature and to make it possible to do
what you want.

We, the Satpy maintainers, expect you to be patient, understanding, and
respectful of both developers and users. Satpy can only be successful if
everyone in the community feels welcome. We also expect you to put in as
much work as you expect out of us. There is no dedicated PyTroll or Satpy
support team, so there may be times when you need to do most of the work
to solve your problem (trying different test cases, environments, etc).

Being respectful includes following the style of the existing code for any
code submissions. Please follow
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines and
limit lines of code to 80 characters whenever possible and when it doesn't
hurt readability. Satpy follows
`Google Style Docstrings <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
for all code API documentation. When in doubt use the existing code as a
guide for how coding should be done.

.. _dev_help:

How do I get help?
==================

The Satpy developers (and all other PyTroll package developers) monitor the:

- `Mailing List <https://groups.google.com/group/pytroll>`_
- `Slack chat <https://pytroll.slack.com/>`_ (get an `invitation <https://pytrollslackin.herokuapp.com/>`_)
- `GitHub issues <https://github.com/pytroll/satpy/issues>`_

How do I submit my changes?
===========================

Any contributions should start with some form of communication (see above) to
let the Satpy maintainers know how you plan to help. The larger the
contribution the more important direct communication is so everyone can avoid
duplicate code and wasted time.
After talking to the Satpy developers any additional work like code or
documentation changes can be provided as a GitHub
`Pull Request <https://help.github.com/articles/creating-a-pull-request/>`_.

To make sure that your code complies with the pytroll python standard, you can
run the `flake8 <http://flake8.pycqa.org/en/latest/>`_ linter on your changes
before you submit them, or even better install a pre-commit hook that runs the
style check for you. To this aim, we provide a configuration file for the
`pre-commit <http://pre-commit.com>`_ tool, that you can install with eg::

  pip install pre-commit
  pre-commit install

running from your base satpy directory. This will automatically check code style for every commit.

Code of Conduct
===============

Satpy follows the same code of conduct as the PyTroll project. For reference
it is copied to this repository in
`CODE_OF_CONDUCT.md <https://github.com/pytroll/satpy/blob/main/CODE_OF_CONDUCT.md>`_.

As stated in the PyTroll home page, this code of conduct applies to the
project space (GitHub) as well as the public space online and offline when
an individual is representing the project or the community. Online examples
of this include the PyTroll Slack team, mailing list, and the PyTroll twitter
account. This code of conduct also applies to in-person situations like
PyTroll Contributor Weeks (PCW), conference meet-ups, or any other time when
the project is being represented.

Any violations of this code of conduct will be handled by the core maintainers
of the project including David Hoese, Martin Raspaud, and Adam Dybbroe.
If you wish to report one of the maintainers for a violation and are
not comfortable with them seeing it, please contact one or more of the other
maintainers to report the violation. Responses to violations will be
determined by the maintainers and may include one or more of the following:

- Verbal warning
- Ask for public apology
- Temporary or permanent ban from in-person events
- Temporary or permanent ban from online communication (Slack, mailing list, etc)

For details see the official
`code of conduct document <https://github.com/pytroll/satpy/blob/main/CODE_OF_CONDUCT.md>`_.
