## Table of Contents

- [Table of Contents](#table-of-contents)
- [Step 1: Build CSP from Source](#step-1-build-csp-from-source)
- [Step 2: Configuring Git and GitHub for Development](#step-2-configuring-git-and-github-for-development)
  - [Create your fork](#create-your-fork)
  - [Configure remotes](#configure-remotes)
  - [Authenticating with GitHub](#authenticating-with-github)
- [Guidelines](#guidelines)

## Step 1: Build CSP from Source

To work on CSP, you are going to need to build it from source. See
[Build CSP from Source](Build-CSP-from-Source) for
detailed build instructions.

Once you've built CSP from a `git` clone, you will also need to
configure `git` and your GitHub account for CSP development.

## Step 2: Configuring Git and GitHub for Development

### Create your fork

The first step is to create a personal fork of CSP. To do so, click
the "fork" button at https://github.com/Point72/csp, or just navigate
[here](https://github.com/Point72/csp/fork) in your browser. Set the
owner of the repository to your personal GitHub account if it is not
already set that way and click "Create fork".

### Configure remotes

Next, you should set some names for the `git` remotes corresponding to
main Point72 repository and your fork. See the [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork) for more information.

### Authenticating with GitHub

If you have not already configured `ssh` access to GitHub, you can find
instructions to do so
[here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh),
including instructions to create an SSH key if you have not done
so. Authenticating with SSH is usually the easiest route. If you are working in
an environment that does not allow SSH connections to GitHub, you can look into
[configuring a hardware
passkey](https://docs.github.com/en/authentication/authenticating-with-a-passkey/about-passkeys)
or adding a [personal access
token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
to avoid the need to type in your password every time you push to your fork.

## Guidelines

After developing a change locally, ensure that both [lints](Build-CSP-from-Source#lint-and-autoformat) and [tests](Build-CSP-from-Source#testing) pass. Commits should be squashed into logical units, and all commits must be signed (e.g. with the `-s` git flag). CSP requires [Developer Certificate of Origin](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin) for all contributions.

If your work is still in-progress, open a [draft pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests). Otherwise, open a normal pull request. It might take a few days for a maintainer to review and provide feedback, so please be patient. If a maintainer asks for changes, please make said changes and squash your commits if necessary. If everything looks good to go, a maintainer will approve and merge your changes for inclusion in the next release.

Please note that non substantive changes, large changes without prior discussion, etc, are not accepted and pull requests may be closed.
