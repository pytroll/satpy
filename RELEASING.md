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

If the current "dev" version is not the desired version run:

```
bumpversion patch
```

Where `patch` is `pre`, `patch`, `minor`, or `major` to get to the correct
version. Check version.py to verify proper version. Then run:

```
bumpversion --tag release
```

To remove the `devN` portion of the version number and tag the release.

8. merge back to master and develop `git merge new_release`
9. remove new_release⁠⁠⁠⁠ branch `git branch -d new_release⁠⁠⁠⁠`
10. push changes to github `git push --follow-tags`
11. Verify travis tests passed and deployed sdist and wheel to PyPI
12. Update version to "dev" version of next release:

```
bumpversion patch
```

Which will set the version to "X.Y.Zdev0"

See [this issue](https://github.com/peritus/bumpversion/issues/77) for more information.

13. Push changes to github `git push` (there shouldn't be any new tags).
