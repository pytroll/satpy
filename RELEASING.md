# Releasing SatPy

prerequisites: `pip install bumpversion setuptools twine`

NB! You do not need `mercurial`. `bumpversion` is supposed to function without it. If it still doesn't work it might be that your PATH variable is screwed up. Check that all elements of your PATH are readable!

1. pull from repo
2. run the unittests
3. checkout master
4. create a branch from there: `git checkout -b new_release`
5. merge develop into it `git merge develop`
6. run `loghub` and update the `CHANGELOG.md` file:

```
loghub pytroll/satpy -u <username> -st v0.8.0 -plg bug "Bugs fixed" -plg enhancement "Features added" -plg documentation "Documentation changes"
```

Don't forget to commit!

7. run `bumpversion` with either `patch`, `minor`, `major`, `pre`, or `num` to reach the desired version number

See [semver.org](http://semver.org/) for the definition of those values.

If the current version number is the "dev" version of the desired version run:

```
# to get to alpha
bumpversion --no-commit --no-tag pre
# to get to beta
bumpversion --no-commit --no-tag pre
# to get to rc
bumpversion --no-commit --no-tag pre
# to get to final release and commit and tag the release
bumpversion pre
```

If the current "dev" version is not the desired version run:

```
bumpversion patch
```

Where `patch` is `patch`, `minor`, or `major`. Check version.py to verify proper version.

8. merge back to master and develop `git merge new_release`
9. remove new_release⁠⁠⁠⁠ branch `git branch -d new_release⁠⁠⁠⁠`
10. push changes to github `git push --follow-tags`
11. Verify travis tests passed and deployed sdist and wheel to PyPI
12. Update version to `dev0` version of next release:

```
bumpversion --no-tag patch
```

Which will set the version to "X.Y.Zdev0"
