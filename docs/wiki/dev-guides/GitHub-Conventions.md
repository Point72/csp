## Triaging Issues

The bug tracker is both a venue for users to communicate to the
developers about defects and a database of known defects. It's up to the
maintainers to ensure issues are high quality.

We have a number of labels that can be applied to sort issues into
categories. If you notice newly created issues that are poorly labeled,
consider adding or removing some labels that do not apply to the issue.

The issue template encourages users to write bug reports that clearly
describe the problem they are having and include steps to reproduce the
issue. However, users sometimes ignore the template or are not used to
GitHub and make mistakes in formatting or communication.

If you are able to infer what they meant and are able to understand the
issue, feel free to edit their issue description to fix formatting or
correct issues with a script demonstrating the issue.

If there is still not enough information or if the issue is unclear,
request more information from the submitter. If they do not respond or
do not clarify sufficiently, close the issue. Try to be polite and have
empathy for inexperienced issue authors.

## How to check out a PR locally

This workflow is described in the [GitHub
docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/checking-out-pull-requests-locally#modifying-an-inactive-pull-request-locally).

1. Identify the pull request ID. This is the number of the pull request
   in the GitHub UI, which shows up in the URL for the pull request. For
   example, https://github.com/Point72/csp/pull/98 has PR ID 98.

1. Fetch the pull request ref and assign it to a local branch name.

   ```bash
   git fetch <remote> pull/<ID>/head:LOCAL_BRANCH_NAME
   ```

   where `<remote>` is the remote (usually `origin`), `<ID>` is the PR ID number and `LOCAL_BRANCH_NAME` is a name
   chosen for the PR branch in your local checkout of CSP.

1. Switch to the PR branch

   ```bash
   git switch LOCAL_BRANCH_NAME
   ```

1. Rebuild CSP

## Pushing Fixups to Pull Requests

Sometimes pull requests don't quite make it across the finish line. In
cases where only a small fixup is required to make a PR mergeable and
the author of the pull request is unresponsive to requests, the best
course of action is often to push to the pull request directly to
resolve the issues.

To do this, check out the pull request locally using the above
instructions. Then make the changes needed for the pull request and push
the local branch back to GitHub:

```bash
git push <remote> LOCAL_BRANCH_NAME
```

where `<remote>` is the remote (usually `origin`) and `LOCAL_BRANCH_NAME` is the name you gave to the PR branch when you
fetched it from GitHub.

Note that if the user who created the pull request selected the option
to forbid pushes to their pull request, you will instead need to
recreate the pull request by pushing the PR branch to your fork and
making a pull request like normal.
