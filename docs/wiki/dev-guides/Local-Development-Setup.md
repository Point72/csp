## Table of Contents

- [Table of Contents](#table-of-contents)
- [Step 1: Build CSP from Source](#step-1-build-csp-from-source)
- [Step 2: Configuring Git and GitHub for Development](#step-2-configuring-git-and-github-for-development)
  - [Create your fork](#create-your-fork)
  - [Configure remotes](#configure-remotes)
  - [Authenticating with GitHub](#authenticating-with-github)
  - [Configure commit signing](#configure-commit-signing)
- [Guidelines](#guidelines)

## Step 1: Build CSP from Source

To work on CSP, you are going to need to build it from source. See
[Build CSP from Source](Build-CSP-from-Source.md) for
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
main Point72 repository and your fork. If you started with a clone of
the main `Point72` repository, you could do something like:

```bash
cd csp
git remote rename origin upstream

# for SSH authentication
git remote add origin git@github.com:<username>/csp.git

# for HTTP authentication
git remote add origin https://github.com/<username>/csp.git
```

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

### Configure commit signing

Additionally, you will need to configure your local `git` setup and
GitHub account to use [commit
signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification). All
commits to the `csp` repository must be signed to increase the
difficulty of a supply-chain attack against the CSP codebase. The
easiest way to do this is to [configure `git` to sign commits with your
SSH
key](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#ssh-commit-signature-verification). You
can also use a [GPG
key](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#gpg-commit-signature-verification)
to sign commits.

In either case, you must also add your public key to your github account
as a signing key. Note that if you have already added an SSH key as an
authentication key, you will need to add it again [as a signing
key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

## Guidelines

After developing a change locally, ensure that both [lints](Build-CSP-from-Source.md#lint-and-autoformat) and [tests](Build-CSP-from-Source.md#testing) pass. Commits should be squashed into logical units, and all commits must be signed (e.g. with the `-s` git flag). CSP requires [Developer Certificate of Origin](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin) for all contributions.

If your work is still in-progress, open a [draft pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests). Otherwise, open a normal pull request. It might take a few days for a maintainer to review and provide feedback, so please be patient. If a maintainer asks for changes, please make said changes and squash your commits if necessary. If everything looks good to go, a maintainer will approve and merge your changes for inclusion in the next release.

Please note that non substantive changes, large changes without prior discussion, etc, are not accepted and pull requests may be closed.
