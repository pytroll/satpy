# Releasing Satpy

1. checkout master
2. pull from repo
3. run the unittests
4. run `loghub`.  Replace <github username> and <previous version> with proper
   values.  To get the previous version run `git tag` and select the most
   recent with highest version number.

```
loghub pytroll/satpy -u <github username> -st v<previous version> -plg bug "Bugs fixed" -plg enhancement "Features added" -plg documentation "Documentation changes" -plg backwards-incompatibility "Backwards incompatible changes"
```

This command will create a CHANGELOG.temp file which need to be added
to the top of the CHANGLOG.md file.  The same content is also printed
to terminal, so that can be copy-pasted, too.  Remember to update also
the version number to the same given in step 5. Don't forget to commit
CHANGELOG.md!

5. Create a tag with the new version number, starting with a 'v', eg:

```
git tag -a v<new version> -m "Version <new version>"
```

For example if the previous tag was `v0.9.0` and the new release is a
patch release, do:

```
git tag -a v0.9.1 -m "Version 0.9.1"
```

See [semver.org](http://semver.org/) on how to write a version number.


6. push changes to github `git push --follow-tags`
7. Verify travis tests passed and deployed sdist and wheel to PyPI
