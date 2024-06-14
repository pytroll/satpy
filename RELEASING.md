# Releasing Satpy

1. checkout main branch
2. pull from repo
3. run the unittests
4. run `loghub` and update the `CHANGELOG.md` file:

   ```
   loghub pytroll/satpy --token $LOGHUB_GITHUB_TOKEN -st $(git tag --sort=-version:refname --list 'v*' | head -n 1) -plg bug "Bugs fixed" -plg enhancement "Features added" -plg documentation "Documentation changes" -plg backwards-incompatibility "Backward incompatible changes" -plg refactor "Refactoring"
   ```

   This uses a `LOGHUB_GITHUB_TOKEN` environment variable. This must be created
   on GitHub and it is recommended that you add it to your `.bashrc` or
   `.bash_profile` or equivalent.

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
7. Verify github action unittests passed.
8. Create a "Release" on GitHub by going to
   https://github.com/pytroll/satpy/releases and clicking "Draft a new release".
   On the next page enter the newly created tag in the "Tag version" field,
   "Version X.Y.Z" in the "Release title" field, and paste the markdown from
   the changelog (the portion under the version section header) in the
   "Describe this release" box. Finally click "Publish release".
9. Verify the GitHub actions for deployment succeed and the release is on PyPI.
