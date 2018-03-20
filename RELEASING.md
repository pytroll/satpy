# Releasing SatPy

prerequisites: `pip install bumpversion setuptools twine`

NB! You do not need `mercurial`. `bumpversion` is supposed to function without it. If it still doesn't work it might be that your PATH variable is screwed up. Check that all elements of your PATH are readable!

1. pull from repo
2. run the unittests
3. checkout master
4. create a branch from there: `git checkout -b new_release`
5. merge develop into it `git merge develop`
6. run `loghub` and update the `CHANGELOG.md` file: `loghub pytroll/satpy -u <username> -st v0.8.0 -lg bug "Bugs fixed" -lg enhancement "Features added" -lg documentation "Documentation changes"`. Don't forget to commit !
7. run `bumpversion` with either patch, minor, major, pre or num (see [semver.org](http://semver.org/) for the definition of those) until reaching the desired version number
8. merge back to master and develop `git merge new_release`
9. remove new_release⁠⁠⁠⁠ branch `git branch -d new_release⁠⁠⁠⁠`
10. push changes to github `git push --follow-tags`
11. Build distribution `python setup.py sdist`
12. push to pypi `twine upload dist/<dist-filename>`. Be sure to remove any possible link to the old pypi in [`~/.pypirc`](https://docs.python.org/3/distutils/packageindex.html#pypirc) by removing the line with `repository = `. Twine will choose the best repository URL for you. You may have to also remove the lines for your username and password.
