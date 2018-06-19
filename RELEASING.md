# Releasing SatPy

prerequisites: `pip install bumpversion setuptools twine`

NB! You do not need `mercurial`. `bumpversion` is supposed to function without it. If it still doesn't work it might be that your PATH variable is screwed up. Check that all elements of your PATH are readable!

1. checkout master
2. pull from repo
3. run the unittests
4. run `loghub` and update the `CHANGELOG.md` file:

```
loghub pytroll/satpy -u <username> -st v0.8.0 -plg bug "Bugs fixed" -plg enhancement "Features added" -plg documentation "Documentation changes"
```

Don't forget to commit!

5. run `bumpversion` with either `patch`, `minor`, `major`, `pre`, or `num` to reach the desired version number

See [semver.org](http://semver.org/) for the definition of those values.

You may need to run bumpversion multiple times in order to reach the desired version. In this case, you can run eg:

```
bumpversion num --no-commit --allow-dirty
```

If the current "dev" version is not the desired version run:

```
bumpversion patch
```

Where `patch` is `pre`, `patch`, `minor`, or `major` to get to the correct
version. Check version.py to verify proper version. Then run:

```
bumpversion --tag release
```

to remove the `devN` portion of the version number and tag the release.

6. push changes to github `git push --follow-tags`
7. Verify travis tests passed and deploy sdist and wheel to PyPI
8. Update version to "dev" version of next release:
```
bumpversion patch
```
Which will set the version to "X.Y.Zdev0"

See [this issue](https://github.com/peritus/bumpversion/issues/77) for more information.

9. Push changes to github `git push` (there shouldn't be any new tags).
