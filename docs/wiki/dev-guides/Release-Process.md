## Table of Contents

- [Table of Contents](#table-of-contents)
- [Doing a "normal" release](#doing-a-normal-release)
  - [Choosing a version number](#choosing-a-version-number)
  - [Preparing and tagging a release](#preparing-and-tagging-a-release)
  - [Releasing to PyPI](#releasing-to-pypi)
    - [A developer's first release](#a-developers-first-release)
    - [Doing the release](#doing-the-release)
    - [Download release artifacts from github actions](#download-release-artifacts-from-github-actions)
    - [Optionally upload to testpypi to test "pip install"](#optionally-upload-to-testpypi-to-test-pip-install)
    - [Upload to pypi](#upload-to-pypi)
  - [Releasing to conda-forge](#releasing-to-conda-forge)
- [Dealing with release mistakes](#dealing-with-release-mistakes)

## Doing a "normal" release

Making regular releases allows users to test code as it gets added and
allows developers to quickly get feedback on changes and new features. A
release announcement is the primary way projects communicate with the
majority of users, so you should think of the release and accompanying
notes as a way to communicate expectations for users.

### Choosing a version number

The clearest signal you can send to users about what to expect from a
release is the version number you choose. Most users expect version
numbers to follow [semantic versioning](https://semver.org/) or semver,
where the version number indicates the sorts of changes to expect.

With semver, There are three kinds of releases, each of which have a
different potential impact on users.

- #### Patch release

  This is the most common kind of release. A patch release should only
  include fixes for bugs or other changes that cannot impact code a
  user writes with the `csp` package. A user should be able to safely
  upgrade CSP from the previous version to a new patch release with
  no changes to the output of their code and no new errors being
  raised, except for fixed bugs. Whether or not a bug fix is
  sufficiently impactful to break backward compatibility is a
  judgement call. It is best to err on the side of safety and do a
  major release if you are unsure.

- #### Minor releases

  This indicates a new feature has been added to the library in a
  backward incompatible. Minor releases can include bugfixes as well
  but should not include backward incompatible changes. For example,
  adding a new keyword argument to a Python function is a backward
  compatible change but removing an argument or changing its name is
  backwards incompatible.

- #### Major releases

  A major release indicates that there are changes in the release that
  are not backward compatible, and users may need to update their code
  to accommodate the change. When possible, efforts should be made to
  communicate to users how to migrate their code.

Note that the primary concern is user impact. Sometimes a bug fix is so
disruptive to users that fixing it qualifies as a major
change. Sometimes a new feature is so big that it fundamentally changes
the nature of the library and it deserves a major release to indicate
that, even if there are no breaking changes. Sometimes a change is
breaking, but only in a very unlikely scenario that can be safely
ignored in practice. It is best to use your good judgement when choosing
a new version number and not slavishly follow rules. Be empathetic to
your users.

### Preparing and tagging a release

Follow these steps when it's time to tag a new release. Before doing
this, you will need to ensure `bump2version` is installed into your
development environment.

> \[!NOTE\]
> If you are working on the main `Point72/csp` repository, set the
> remote as `origin` in all git commands. If you are working off a personal fork,
> use the corresponding remote of the main repository (e.g. `upstream`).
> Ensure tags are still pushed to `Point72/csp` directly.

1. Ensure your local clone of `csp` is synced up with GitHub, including
   any tags that have been pushed since you last synced:

   ```bash
   git pull <remote> main --tags
   ```

1. Make a branch and update version numbers in your local clone using
   the `bump2version` integration in the Makefile.

   First, make a branch that will be pushed to the main `csp`
   repository. Using a name like `release/v0.3.4` should avoid any
   conflicts and make it clear what the branch is for.

   ```bash
   git checkout -b release/v0.x.x
   ```

   You can update the version number in the codebase using `bump2version`. For a
   bugfix release, you would do:

   ```bash
   make patch
   ```

   Similarly, `make minor` and `make major` will update the version
   numbers for minor and major releases, respectively. Double-check
   that the version numbers have been updated correctly with
   `git diff`, and then `git commit -s` the change.

1. Push your branch to GitHub, and trigger a "full" test run on the branch.

   Navigate to the [GitHub Actions "build status"
   workflow](https://github.com/Point72/csp/actions/workflows/build.yml). Click
   the white "Run workflow" button, make sure the "Run full CI" radio
   button is selected, and click the green "Run workflow" button to
   launch the test run.

1. Propose a pull request from the branch containing the version number updated.
   Add a link to the successful full test run for reviewers.

1. Review and merge the pull request. Make sure you delete the branch
   afterwards.

1. Tag the release

   Use the version number `bump2version` generated and make sure the
   tag name begins with a `v`.

   ```bash
   git tag v0.2.0
   ```

1. Push the tag to GitHub using the tag name created in the previous step.

   ```bash
   git push <remote> tag v0.2.0
   ```

   You will need access in the repository settings to be able to push
   tags to the repository. Access to merge pull requests is not
   sufficient to overcome the tag protection settings in the repository.

Pushing a tag name that begins with `v` should automatically trigger a
full test run that will generate a GitHub release. You can follow along
with this run in the GitHub actions interface. There should be two
actions running, one for the push to `main` and one for the new tag. You
want to inspect the action running for the new tag. Once the run
finishes, there should be a new release on the ["Releases"
page](https://github.com/Point72/csp/releases).
If the release is in "Draft" state, click on the pencil icon to
"Edit" and publish it with the "Publish release" button.

### Releasing to PyPI

#### A developer's first release

If this is your first release, you will need an account on pypi.org and
your account will need to be added as a maintainer to the CSP project
on PyPI. You will also need to have two factor authentication enabled on
your PyPI account.

Once that is set up, navigate to the API token page in your PyPI
settings and generate an API token scoped to the CSP project. **Do not**
navigate away from the page displaying the API token before the next
step.

Optionally, you can create an account on test.pypi.org if you would like
to do a dry run of the release. You will also need to set up an API
token on test.pypi.org.

Create a `.pypirc` file in your home directory, and add the following
content:

```
[distutils]
  index-servers =
    testpypi
    csp
[testpypi]
  username = __token__
  password = <your API key, including pypi- prefix>
[csp]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = <your API key, including pypi- prefix>
```

#### Doing the release

#### Download release artifacts from github actions

Make sure you are in the root of the `csp` repository and execute the
following commands.

```bash
rm -r dist
mkdir dist
cd dist
curl -sL https://api.github.com/repos/Point72/csp/releases/latest | grep browser_download_url | cut -d '"' -f 4 | xargs -I '{}' curl -LO {}
cd ..
```

You should end up with a copy of the wheel files and source distribution
associated with the GitHub release. You should verify that all files
were successfully downloaded. Currently there are 6 MacOS wheels and 4
linux wheels. There should only be one source distribution.

Optionally, you can lint the release artifacts with

```bash=
twine check --strict dist/*
```

This happens as part of the CI so this should only be a double-check.

#### Optionally upload to testpypi to test "pip install"

```
twine upload --repository testpypi dist/*
```

You can check that the wheel installs correctly with `pip` from
`testpypi` like so:

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple csp
```

Note that `extra-index-url` is necessary to ensure downloading
dependencies succeeds.

#### Upload to pypi

If you are sure the release is ready, you can upload to pypi like so:

```bash
twine upload --repository csp dist/*
```

Note that this assumes you have a `.pypirc` set up as explained above.

Assuming the upload completes successfully, you should see a message
like

```
View at:
https://pypi.org/project/csp/<version number>/
```

### Releasing to conda-forge

The `conda-forge` release process is largely automated. Maintainers who
are listed under the `extra.recipe-maintainers` field in the `csp`
recipe hosted in [the conda-forge feedstock
repository](https://github.com/conda-forge/csp-feedstock/blob/main/recipe/meta.yaml)
should be automatically subscribed to notifications for the
repository. The [conda-forge maintainer documentation](https://conda-forge.org/docs/maintainer/updating_pkgs/)
has the relevant guidelines and procedures, and should be read thoroughly before
making any changes.

## Dealing with release mistakes

Sometimes releases go wrong. Here is what to do when that happens. This
[blog post from Brett
Cannon](https://snarky.ca/what-to-do-when-you-botch-a-release-on-pypi/)
covers how to deal with various kinds of release mistakes on PyPI.
Some things to remember after reading that post:

- Completely broken releases should be yanked.
  - A yanked release is *not permanently deleted*
- Problems with metadata (e.g. the readme or metadata in
  `pyproject.toml`) can be dealt with by creating a `post` release.
- A problem with a wheel binary can be fixed by simply replacing the
  wheel file with a new wheel that has an incremented [build
  number](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention)
  with `twine upload`.
- If private data is published accidentally, a release can be
  permanently deleted. This should only be used as a last resort option.
